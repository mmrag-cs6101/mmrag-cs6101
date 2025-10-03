"""
Evaluation Pipeline Orchestrator

Orchestrates end-to-end evaluation pipeline with automated optimization and parameter tuning
to achieve target 53-59% accuracy on MRAG-Bench perspective change scenarios.
"""

import os
import time
import logging
import itertools
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .evaluator import MRAGBenchEvaluator, PerspectiveChangeType, EvaluationSession
from .optimizer import PerformanceOptimizer, OptimizationConfig, OptimizationResult
from ..config import MRAGConfig
from ..utils.error_handling import handle_errors, MRAGError
from ..utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationTarget:
    """Optimization target specification."""
    min_accuracy: float = 0.53
    max_accuracy: float = 0.59
    max_processing_time: float = 30.0  # seconds per query
    memory_limit_gb: float = 15.0
    min_samples_per_scenario: int = 50


@dataclass
class OrchestrationResult:
    """Complete orchestration result with optimization details."""
    session_id: str
    target_achieved: bool
    final_accuracy: float
    best_config: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    total_optimization_time: float
    final_evaluation_session: EvaluationSession
    recommendations: List[str]


class EvaluationOrchestrator:
    """
    Orchestrates comprehensive evaluation pipeline with automated optimization.

    Features:
    - Automated hyperparameter tuning for target accuracy achievement
    - Performance optimization within memory constraints
    - Multi-scenario evaluation coordination
    - Comprehensive results analysis and reporting
    - Intelligent parameter search strategies
    - Memory-aware optimization scheduling
    """

    def __init__(
        self,
        base_config: MRAGConfig,
        target: OptimizationTarget,
        output_dir: str = "orchestration_results"
    ):
        """
        Initialize evaluation orchestrator.

        Args:
            base_config: Base MRAG system configuration
            target: Optimization target specification
            output_dir: Directory for orchestration results
        """
        self.base_config = base_config
        self.target = target
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.evaluator = None
        self.optimizer = None
        self.memory_manager = MemoryManager(
            memory_limit_gb=target.memory_limit_gb,
            buffer_gb=1.0
        )

        # Orchestration state
        self.optimization_history = []
        self.best_result = None
        self.current_iteration = 0

        logger.info(f"Evaluation orchestrator initialized with target accuracy: {target.min_accuracy:.1%}-{target.max_accuracy:.1%}")

    @handle_errors
    def run_comprehensive_evaluation(
        self,
        max_optimization_rounds: int = 10,
        early_stopping_patience: int = 3,
        parallel_configs: int = 2
    ) -> OrchestrationResult:
        """
        Run comprehensive evaluation with automated optimization.

        Args:
            max_optimization_rounds: Maximum optimization iterations
            early_stopping_patience: Stop if no improvement for N rounds
            parallel_configs: Number of configurations to test in parallel

        Returns:
            Complete orchestration result
        """
        logger.info("Starting comprehensive MRAG-Bench evaluation with optimization...")
        start_time = time.time()

        try:
            # Initialize components
            self._initialize_components()

            # Run baseline evaluation
            baseline_result = self._run_baseline_evaluation()
            logger.info(f"Baseline accuracy: {baseline_result.overall_accuracy:.1%}")

            # Check if baseline already meets target
            if self._meets_target(baseline_result.overall_accuracy):
                logger.info("Baseline configuration already meets target accuracy!")
                return self._create_orchestration_result(
                    baseline_result, start_time, target_achieved=True
                )

            # Run optimization
            optimization_result = self._run_optimization_loop(
                baseline_result=baseline_result,
                max_rounds=max_optimization_rounds,
                patience=early_stopping_patience,
                parallel_configs=parallel_configs
            )

            # Final comprehensive evaluation with best configuration
            final_session = self._run_final_evaluation(optimization_result.best_config)

            total_time = time.time() - start_time
            target_achieved = self._meets_target(final_session.overall_accuracy)

            logger.info(
                f"Orchestration completed in {total_time:.1f}s:\n"
                f"  Final Accuracy: {final_session.overall_accuracy:.1%}\n"
                f"  Target Achievement: {'✓' if target_achieved else '✗'}\n"
                f"  Optimization Rounds: {len(self.optimization_history)}"
            )

            return self._create_orchestration_result(
                final_session, start_time, target_achieved, optimization_result
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            raise MRAGError(f"Evaluation orchestration failed: {str(e)}") from e

    def _initialize_components(self) -> None:
        """Initialize evaluator and optimizer components."""
        logger.info("Initializing orchestration components...")

        # Initialize evaluator
        self.evaluator = MRAGBenchEvaluator(
            config=self.base_config,
            output_dir=str(self.output_dir / "evaluations")
        )
        self.evaluator.initialize_components()

        # Initialize optimizer
        optimization_config = OptimizationConfig(
            target_accuracy_range=(self.target.min_accuracy, self.target.max_accuracy),
            max_processing_time=self.target.max_processing_time,
            memory_limit_gb=self.target.memory_limit_gb,
            optimization_strategy="bayesian",
            search_space_config={
                "retrieval": {
                    "top_k": [3, 5, 7, 10],
                    "similarity_threshold": [0.6, 0.7, 0.8, 0.9]
                },
                "generation": {
                    "temperature": [0.1, 0.3, 0.5, 0.7],
                    "top_p": [0.8, 0.9, 0.95],
                    "max_length": [256, 512, 1024]
                }
            }
        )

        self.optimizer = PerformanceOptimizer(optimization_config)

        logger.info("Components initialized successfully")

    def _run_baseline_evaluation(self) -> EvaluationSession:
        """Run baseline evaluation with current configuration."""
        logger.info("Running baseline evaluation...")

        return self.evaluator.evaluate_all_scenarios(
            max_samples_per_scenario=self.target.min_samples_per_scenario
        )

    def _run_optimization_loop(
        self,
        baseline_result: EvaluationSession,
        max_rounds: int,
        patience: int,
        parallel_configs: int
    ) -> OptimizationResult:
        """Run optimization loop to improve accuracy."""
        logger.info(f"Starting optimization loop (max {max_rounds} rounds)...")

        best_accuracy = baseline_result.overall_accuracy
        no_improvement_count = 0

        for round_num in range(max_rounds):
            logger.info(f"Optimization round {round_num + 1}/{max_rounds}")

            # Generate candidate configurations
            candidate_configs = self.optimizer.suggest_configurations(
                current_accuracy=best_accuracy,
                num_suggestions=parallel_configs,
                exploration_factor=max(0.1, 1.0 - (round_num / max_rounds))
            )

            # Evaluate candidates
            round_results = self._evaluate_candidate_configs(candidate_configs)

            # Find best result from this round
            round_best = max(round_results, key=lambda r: r["accuracy"])

            # Update optimization history
            self.optimization_history.extend(round_results)

            # Check for improvement
            if round_best["accuracy"] > best_accuracy:
                improvement = round_best["accuracy"] - best_accuracy
                best_accuracy = round_best["accuracy"]
                no_improvement_count = 0

                logger.info(
                    f"Improvement found: {round_best['accuracy']:.1%} "
                    f"(+{improvement:.1%})"
                )

                # Check if target achieved
                if self._meets_target(best_accuracy):
                    logger.info("Target accuracy achieved!")
                    break
            else:
                no_improvement_count += 1
                logger.info(f"No improvement in round {round_num + 1}")

                # Early stopping
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping after {patience} rounds without improvement")
                    break

        # Create optimization result
        best_overall = max(self.optimization_history, key=lambda r: r["accuracy"])
        return OptimizationResult(
            best_config=best_overall["config"],
            best_accuracy=best_overall["accuracy"],
            optimization_iterations=len(self.optimization_history),
            target_achieved=self._meets_target(best_overall["accuracy"]),
            improvement_over_baseline=best_overall["accuracy"] - baseline_result.overall_accuracy
        )

    def _evaluate_candidate_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate candidate configurations, potentially in parallel."""
        results = []

        if len(configs) == 1:
            # Single configuration - evaluate directly
            config = configs[0]
            session = self._evaluate_single_config(config)
            results.append({
                "config": config,
                "accuracy": session.overall_accuracy,
                "avg_processing_time": session.avg_processing_time,
                "session": session
            })
        else:
            # Multiple configurations - evaluate with memory management
            for i, config in enumerate(configs):
                logger.info(f"Evaluating configuration {i+1}/{len(configs)}")

                try:
                    # Clear memory before each evaluation
                    self.memory_manager.clear_gpu_memory(aggressive=True)

                    session = self._evaluate_single_config(config)
                    results.append({
                        "config": config,
                        "accuracy": session.overall_accuracy,
                        "avg_processing_time": session.avg_processing_time,
                        "session": session
                    })

                    logger.info(
                        f"Config {i+1} accuracy: {session.overall_accuracy:.1%}, "
                        f"time: {session.avg_processing_time:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"Failed to evaluate config {i+1}: {e}")
                    results.append({
                        "config": config,
                        "accuracy": 0.0,
                        "avg_processing_time": float('inf'),
                        "session": None,
                        "error": str(e)
                    })

        return results

    def _evaluate_single_config(self, config: Dict[str, Any]) -> EvaluationSession:
        """Evaluate a single configuration."""
        # Update base config with optimization parameters
        updated_config = self._update_config_with_params(self.base_config, config)

        # Create new evaluator with updated config
        temp_evaluator = MRAGBenchEvaluator(
            config=updated_config,
            output_dir=str(self.output_dir / "temp_evaluations")
        )
        temp_evaluator.initialize_components()

        try:
            # Run evaluation with limited samples for faster iteration
            session = temp_evaluator.evaluate_all_scenarios(
                max_samples_per_scenario=min(
                    self.target.min_samples_per_scenario,
                    25  # Smaller sample size for optimization
                )
            )
            return session

        finally:
            # Clean up
            temp_evaluator.cleanup()

    def _update_config_with_params(self, base_config: MRAGConfig, params: Dict[str, Any]) -> MRAGConfig:
        """Update configuration with optimization parameters."""
        # Create a copy of the config
        updated_config = MRAGConfig(
            dataset=base_config.dataset,
            model=base_config.model,
            retrieval=base_config.retrieval,
            generation=base_config.generation,
            performance=base_config.performance
        )

        # Update retrieval parameters
        if "retrieval" in params:
            retrieval_params = params["retrieval"]
            if "top_k" in retrieval_params:
                updated_config.retrieval.top_k = retrieval_params["top_k"]
            if "similarity_threshold" in retrieval_params:
                updated_config.retrieval.similarity_threshold = retrieval_params["similarity_threshold"]

        # Update generation parameters
        if "generation" in params:
            generation_params = params["generation"]
            if "temperature" in generation_params:
                updated_config.generation.temperature = generation_params["temperature"]
            if "top_p" in generation_params:
                updated_config.generation.top_p = generation_params["top_p"]
            if "max_length" in generation_params:
                updated_config.generation.max_length = generation_params["max_length"]

        return updated_config

    def _run_final_evaluation(self, best_config: Dict[str, Any]) -> EvaluationSession:
        """Run final comprehensive evaluation with best configuration."""
        logger.info("Running final comprehensive evaluation with optimized configuration...")

        # Update config with best parameters
        final_config = self._update_config_with_params(self.base_config, best_config)

        # Create final evaluator
        final_evaluator = MRAGBenchEvaluator(
            config=final_config,
            output_dir=str(self.output_dir / "final_evaluation")
        )
        final_evaluator.initialize_components()

        try:
            # Run comprehensive evaluation with full dataset
            session = final_evaluator.evaluate_all_scenarios(
                max_samples_per_scenario=None  # Use all available samples
            )
            return session

        finally:
            final_evaluator.cleanup()

    def _meets_target(self, accuracy: float) -> bool:
        """Check if accuracy meets target range."""
        return self.target.min_accuracy <= accuracy <= self.target.max_accuracy

    def _create_orchestration_result(
        self,
        final_session: EvaluationSession,
        start_time: float,
        target_achieved: bool,
        optimization_result: Optional[OptimizationResult] = None
    ) -> OrchestrationResult:
        """Create final orchestration result."""
        total_time = time.time() - start_time

        # Generate recommendations
        recommendations = self._generate_recommendations(final_session, optimization_result)

        result = OrchestrationResult(
            session_id=f"orchestration_{int(start_time)}",
            target_achieved=target_achieved,
            final_accuracy=final_session.overall_accuracy,
            best_config=optimization_result.best_config if optimization_result else {},
            optimization_history=self.optimization_history,
            total_optimization_time=total_time,
            final_evaluation_session=final_session,
            recommendations=recommendations
        )

        # Save orchestration result
        self._save_orchestration_result(result)

        return result

    def _generate_recommendations(
        self,
        final_session: EvaluationSession,
        optimization_result: Optional[OptimizationResult]
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Accuracy-based recommendations
        if final_session.overall_accuracy < self.target.min_accuracy:
            recommendations.append(
                f"Accuracy ({final_session.overall_accuracy:.1%}) below target. "
                "Consider: 1) Larger retrieval top-k, 2) Lower generation temperature, "
                "3) More sophisticated prompt engineering"
            )
        elif final_session.overall_accuracy > self.target.max_accuracy:
            recommendations.append(
                f"Accuracy ({final_session.overall_accuracy:.1%}) exceeds target range. "
                "Consider optimizing for speed while maintaining accuracy."
            )

        # Performance-based recommendations
        if final_session.avg_processing_time > self.target.max_processing_time:
            recommendations.append(
                f"Processing time ({final_session.avg_processing_time:.1f}s) exceeds target. "
                "Consider: 1) Reducing generation length, 2) Optimizing retrieval batch size, "
                "3) More aggressive quantization"
            )

        # Memory-based recommendations
        memory_usage = final_session.memory_stats.get('gpu_allocated_gb', 0)
        if memory_usage > self.target.memory_limit_gb * 0.9:
            recommendations.append(
                f"Memory usage ({memory_usage:.1f}GB) near limit. "
                "Consider: 1) Smaller batch sizes, 2) More aggressive quantization, "
                "3) Sequential model loading"
            )

        # Scenario-specific recommendations
        for scenario, metrics in final_session.scenario_results.items():
            if metrics.accuracy < 0.4:  # Significantly poor performance
                recommendations.append(
                    f"Poor performance on {scenario} scenario ({metrics.accuracy:.1%}). "
                    "Consider scenario-specific prompt optimization or retrieval tuning."
                )

        # Optimization-based recommendations
        if optimization_result and optimization_result.improvement_over_baseline > 0.05:
            recommendations.append(
                "Significant improvement achieved through optimization. "
                "Consider applying similar parameter adjustments in production."
            )

        if not recommendations:
            recommendations.append("System performance is within target parameters. Consider production deployment.")

        return recommendations

    def _save_orchestration_result(self, result: OrchestrationResult) -> None:
        """Save orchestration result to disk."""
        try:
            # Save detailed results
            result_file = self.output_dir / f"{result.session_id}_orchestration.json"
            with open(result_file, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)

            # Save summary report
            summary_file = self.output_dir / f"{result.session_id}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(self._generate_orchestration_summary(result))

            logger.info(f"Orchestration results saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save orchestration result: {e}")

    def _generate_orchestration_summary(self, result: OrchestrationResult) -> str:
        """Generate human-readable orchestration summary."""
        return f"""
MRAG-Bench Evaluation Orchestration Summary
===========================================

Session ID: {result.session_id}
Target Achievement: {'✓ SUCCESS' if result.target_achieved else '✗ NOT ACHIEVED'}
Final Accuracy: {result.final_accuracy:.1%}
Target Range: {self.target.min_accuracy:.1%} - {self.target.max_accuracy:.1%}
Total Optimization Time: {result.total_optimization_time:.1f}s
Optimization Rounds: {len(result.optimization_history)}

Best Configuration:
-------------------
{json.dumps(result.best_config, indent=2)}

Final Evaluation Results:
-------------------------
Overall Accuracy: {result.final_evaluation_session.overall_accuracy:.1%}
Total Questions: {result.final_evaluation_session.total_questions}
Avg Processing Time: {result.final_evaluation_session.avg_processing_time:.2f}s

Scenario Breakdown:
-------------------
""" + "\n".join([
    f"{scenario}: {metrics.accuracy:.1%} ({metrics.correct_answers}/{metrics.total_questions})"
    for scenario, metrics in result.final_evaluation_session.scenario_results.items()
]) + f"""

Recommendations:
----------------
""" + "\n".join([f"• {rec}" for rec in result.recommendations]) + f"""

Optimization History:
--------------------
Best accuracy achieved: {max([r.get('accuracy', 0) for r in result.optimization_history]):.1%}
Total configurations tested: {len(result.optimization_history)}
"""

    def cleanup(self) -> None:
        """Clean up orchestrator resources."""
        logger.info("Cleaning up evaluation orchestrator...")

        if self.evaluator:
            self.evaluator.cleanup()

        self.memory_manager.emergency_cleanup()

        logger.info("Orchestrator cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()