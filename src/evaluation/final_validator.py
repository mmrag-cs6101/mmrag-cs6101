"""
Sprint 10: Final Accuracy Validation Framework

Comprehensive final validation system for MRAG-Bench accuracy measurement with:
- Multi-run validation for statistical confidence
- Complete evaluation on all 778 perspective change samples
- Statistical significance testing and confidence intervals
- Comprehensive system verification and performance validation
"""

import os
import time
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Try to import scipy for advanced statistics
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scipy not available - using fallback statistical methods")

from .evaluator import MRAGBenchEvaluator, PerspectiveChangeType, ScenarioMetrics
from ..config import MRAGConfig
from ..utils.error_handling import handle_errors, MRAGError
from ..utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class MultiRunStatistics:
    """Statistical analysis across multiple evaluation runs."""
    mean_accuracy: float
    std_accuracy: float
    median_accuracy: float
    min_accuracy: float
    max_accuracy: float
    confidence_interval_95: Tuple[float, float]
    coefficient_of_variation: float
    individual_run_accuracies: List[float]
    statistical_significance: bool
    p_value: Optional[float] = None


@dataclass
class ScenarioFinalResults:
    """Final validation results for a single scenario."""
    scenario_type: str
    total_samples: int
    correct_answers: int
    accuracy: float
    confidence_interval_95: Tuple[float, float]

    # Multi-run statistics
    multi_run_stats: Optional[MultiRunStatistics]

    # Performance metrics
    avg_processing_time: float
    avg_retrieval_time: float
    avg_generation_time: float

    # Quality metrics
    avg_confidence_score: float
    std_confidence_score: float
    error_rate: float

    # Target validation
    in_target_range: bool
    target_range: Tuple[float, float]


@dataclass
class SystemPerformanceMetrics:
    """Comprehensive system performance validation."""
    # Timing metrics
    avg_query_time: float
    p50_query_time: float
    p95_query_time: float
    p99_query_time: float
    total_evaluation_time: float

    # Memory metrics
    peak_memory_gb: float
    avg_memory_gb: float
    memory_utilization_percent: float
    memory_within_limit: bool

    # Reliability metrics
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    error_rate: float

    # Throughput metrics
    queries_per_second: float
    samples_per_minute: float


@dataclass
class FinalValidationResults:
    """Complete Sprint 10 final validation results."""
    # Overall accuracy metrics
    overall_accuracy: float
    overall_confidence_interval: Tuple[float, float]
    total_questions: int
    total_correct: int
    target_achieved: bool
    target_range: Tuple[float, float]

    # Per-scenario results
    scenario_results: Dict[str, ScenarioFinalResults]
    scenarios_in_target: int
    scenario_consistency: float

    # Multi-run validation
    num_evaluation_runs: int
    multi_run_statistics: Optional[MultiRunStatistics]
    cross_run_consistency: float

    # System performance
    performance_metrics: SystemPerformanceMetrics

    # Statistical validation
    statistical_confidence: str  # "high", "medium", "low"
    baseline_comparison: Optional[Dict[str, Any]]

    # Recommendations
    recommendations: List[str]
    production_readiness: str  # "ready", "needs_optimization", "not_ready"

    # Metadata
    timestamp: str
    total_validation_time: float
    configuration_summary: Dict[str, Any]


class FinalAccuracyValidator:
    """
    Final accuracy validation system for Sprint 10.

    Provides comprehensive accuracy validation with:
    - Complete evaluation on all 778 perspective change samples
    - Multi-run validation for statistical confidence
    - Detailed statistical analysis and significance testing
    - Performance and reliability verification
    - Production readiness assessment
    """

    def __init__(
        self,
        config: MRAGConfig,
        target_range: Tuple[float, float] = (0.53, 0.59),
        output_dir: str = "output/sprint10"
    ):
        """
        Initialize final accuracy validator.

        Args:
            config: MRAG system configuration
            target_range: Target accuracy range (min, max)
            output_dir: Output directory for results
        """
        self.config = config
        self.target_range = target_range
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.evaluator = MRAGBenchEvaluator(
            config=config,
            output_dir=str(self.output_dir / "evaluations")
        )

        self.memory_manager = MemoryManager(
            memory_limit_gb=config.performance.memory_limit_gb,
            buffer_gb=config.performance.memory_buffer_gb
        )

        # Validation state
        self.evaluation_runs = []
        self.performance_history = []

        logger.info(
            f"Final Accuracy Validator initialized - "
            f"Target range: {target_range[0]:.1%}-{target_range[1]:.1%}"
        )

    @handle_errors
    def run_comprehensive_validation(
        self,
        num_runs: int = 3,
        full_dataset: bool = True,
        max_samples_per_scenario: Optional[int] = None
    ) -> FinalValidationResults:
        """
        Run comprehensive final validation across all scenarios.

        Args:
            num_runs: Number of evaluation runs for statistical confidence
            full_dataset: Use complete dataset (all 778 samples)
            max_samples_per_scenario: Limit samples per scenario (overrides full_dataset)

        Returns:
            Complete final validation results
        """
        logger.info(
            f"Starting Sprint 10 Final Accuracy Validation\n"
            f"  Target Range: {self.target_range[0]:.1%} - {self.target_range[1]:.1%}\n"
            f"  Evaluation Runs: {num_runs}\n"
            f"  Full Dataset: {full_dataset}"
        )

        start_time = time.time()

        try:
            # Initialize evaluation components
            self.evaluator.initialize_components()

            # Determine sample limits
            if max_samples_per_scenario is None and full_dataset:
                # Use complete dataset - all 778 samples distributed as:
                # angle: ~322, partial: ~246, scope: ~102, occlusion: ~108
                max_samples_per_scenario = None

            # Run multiple evaluation runs for statistical confidence
            if num_runs > 1:
                logger.info(f"Performing {num_runs} evaluation runs for statistical validation...")
                run_results = self._run_multiple_evaluations(
                    num_runs=num_runs,
                    max_samples_per_scenario=max_samples_per_scenario
                )
            else:
                logger.info("Performing single comprehensive evaluation...")
                run_results = [self._run_single_evaluation(max_samples_per_scenario)]

            # Aggregate results across runs
            final_results = self._aggregate_multi_run_results(run_results)

            # Collect performance metrics
            performance_metrics = self._collect_performance_metrics(run_results)
            final_results.performance_metrics = performance_metrics

            # Statistical validation
            statistical_confidence = self._assess_statistical_confidence(final_results)
            final_results.statistical_confidence = statistical_confidence

            # Production readiness assessment
            production_readiness = self._assess_production_readiness(final_results)
            final_results.production_readiness = production_readiness

            # Generate recommendations
            recommendations = self._generate_final_recommendations(final_results)
            final_results.recommendations = recommendations

            # Set metadata
            final_results.timestamp = datetime.now().isoformat()
            final_results.total_validation_time = time.time() - start_time
            final_results.configuration_summary = self._get_config_summary()

            # Save results
            self._save_final_results(final_results)

            # Log summary
            self._log_validation_summary(final_results)

            return final_results

        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            raise MRAGError(f"Sprint 10 final validation failed: {str(e)}") from e

    def _run_single_evaluation(
        self,
        max_samples_per_scenario: Optional[int]
    ) -> Dict[str, Any]:
        """Run single comprehensive evaluation across all scenarios."""
        logger.info("Running comprehensive evaluation across all 4 scenarios...")

        start_time = time.time()
        scenario_results = {}
        total_questions = 0
        total_correct = 0
        all_processing_times = []
        all_retrieval_times = []
        all_generation_times = []
        all_confidence_scores = []

        # Evaluate each perspective change scenario
        for scenario_type in PerspectiveChangeType:
            try:
                logger.info(f"Evaluating {scenario_type.value} scenario...")

                metrics = self.evaluator.evaluate_scenario(
                    scenario_type=scenario_type,
                    max_samples=max_samples_per_scenario,
                    use_cache=False  # Always fresh evaluation
                )

                scenario_results[scenario_type.value] = metrics
                total_questions += metrics.total_questions
                total_correct += metrics.correct_answers

                # Collect timing and quality metrics
                all_processing_times.extend([metrics.avg_processing_time] * metrics.total_questions)
                all_retrieval_times.extend([metrics.avg_retrieval_time] * metrics.total_questions)
                all_generation_times.extend([metrics.avg_generation_time] * metrics.total_questions)
                all_confidence_scores.extend(metrics.confidence_scores)

                logger.info(
                    f"  {scenario_type.value}: {metrics.accuracy:.1%} "
                    f"({metrics.correct_answers}/{metrics.total_questions})"
                )

            except Exception as e:
                logger.error(f"Failed to evaluate {scenario_type.value}: {e}")
                raise

        # Calculate overall metrics
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        evaluation_time = time.time() - start_time

        # Get memory statistics
        memory_stats = self.memory_manager.monitor.get_current_stats()

        return {
            "scenario_results": scenario_results,
            "overall_accuracy": overall_accuracy,
            "total_questions": total_questions,
            "total_correct": total_correct,
            "evaluation_time": evaluation_time,
            "avg_processing_time": np.mean(all_processing_times) if all_processing_times else 0.0,
            "avg_retrieval_time": np.mean(all_retrieval_times) if all_retrieval_times else 0.0,
            "avg_generation_time": np.mean(all_generation_times) if all_generation_times else 0.0,
            "avg_confidence_score": np.mean(all_confidence_scores) if all_confidence_scores else 0.0,
            "peak_memory_gb": memory_stats.gpu_allocated_gb,
            "processing_times": all_processing_times,
            "confidence_scores": all_confidence_scores
        }

    def _run_multiple_evaluations(
        self,
        num_runs: int,
        max_samples_per_scenario: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Run multiple evaluation runs for statistical validation."""
        run_results = []

        for run_idx in range(num_runs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluation Run {run_idx + 1}/{num_runs}")
            logger.info(f"{'='*60}")

            # Clear memory before each run
            self.memory_manager.clear_gpu_memory(aggressive=True)

            # Run evaluation
            run_result = self._run_single_evaluation(max_samples_per_scenario)
            run_result["run_number"] = run_idx + 1
            run_results.append(run_result)

            logger.info(
                f"Run {run_idx + 1} Complete - "
                f"Accuracy: {run_result['overall_accuracy']:.1%}"
            )

        return run_results

    def _aggregate_multi_run_results(
        self,
        run_results: List[Dict[str, Any]]
    ) -> FinalValidationResults:
        """Aggregate results from multiple evaluation runs."""
        logger.info("Aggregating multi-run results...")

        # Use the most recent run as base
        base_run = run_results[-1]

        # Aggregate per-scenario results
        scenario_final_results = {}
        for scenario_name in base_run["scenario_results"].keys():
            scenario_metrics = [
                run["scenario_results"][scenario_name]
                for run in run_results
            ]

            scenario_final_results[scenario_name] = self._create_scenario_final_result(
                scenario_name,
                scenario_metrics,
                len(run_results)
            )

        # Calculate overall metrics
        overall_accuracies = [run["overall_accuracy"] for run in run_results]
        overall_accuracy = np.mean(overall_accuracies)

        # Overall confidence interval (Wilson score or normal approximation)
        total_questions = base_run["total_questions"]
        overall_ci = self._calculate_confidence_interval(
            overall_accuracy,
            total_questions
        )

        # Multi-run statistics (if multiple runs)
        if len(run_results) > 1:
            multi_run_stats = self._calculate_multi_run_statistics(
                overall_accuracies,
                self.target_range
            )
            cross_run_consistency = self._calculate_consistency(overall_accuracies)
        else:
            multi_run_stats = None
            cross_run_consistency = 1.0

        # Determine target achievement
        target_achieved = self.target_range[0] <= overall_accuracy <= self.target_range[1]

        # Count scenarios in target
        scenarios_in_target = sum(
            1 for result in scenario_final_results.values()
            if result.in_target_range
        )

        # Scenario consistency (variance across scenarios)
        scenario_accuracies = [
            result.accuracy for result in scenario_final_results.values()
        ]
        scenario_consistency = self._calculate_consistency(scenario_accuracies)

        # Create final results structure (performance metrics added later)
        return FinalValidationResults(
            overall_accuracy=overall_accuracy,
            overall_confidence_interval=overall_ci,
            total_questions=total_questions,
            total_correct=int(overall_accuracy * total_questions),
            target_achieved=target_achieved,
            target_range=self.target_range,
            scenario_results=scenario_final_results,
            scenarios_in_target=scenarios_in_target,
            scenario_consistency=scenario_consistency,
            num_evaluation_runs=len(run_results),
            multi_run_statistics=multi_run_stats,
            cross_run_consistency=cross_run_consistency,
            performance_metrics=None,  # Set later
            statistical_confidence="",  # Set later
            baseline_comparison=None,
            recommendations=[],  # Set later
            production_readiness="",  # Set later
            timestamp="",  # Set later
            total_validation_time=0.0,  # Set later
            configuration_summary={}  # Set later
        )

    def _create_scenario_final_result(
        self,
        scenario_name: str,
        scenario_metrics_list: List[ScenarioMetrics],
        num_runs: int
    ) -> ScenarioFinalResults:
        """Create final result for a single scenario from multiple runs."""
        # Use most recent metrics as base
        base_metrics = scenario_metrics_list[-1]

        # Calculate multi-run statistics if multiple runs
        if num_runs > 1:
            accuracies = [m.accuracy for m in scenario_metrics_list]
            multi_run_stats = self._calculate_multi_run_statistics(
                accuracies,
                self.target_range
            )
        else:
            multi_run_stats = None

        # Average metrics across runs
        avg_accuracy = np.mean([m.accuracy for m in scenario_metrics_list])
        avg_processing_time = np.mean([m.avg_processing_time for m in scenario_metrics_list])
        avg_retrieval_time = np.mean([m.avg_retrieval_time for m in scenario_metrics_list])
        avg_generation_time = np.mean([m.avg_generation_time for m in scenario_metrics_list])

        # Aggregate confidence scores
        all_confidence_scores = []
        for m in scenario_metrics_list:
            all_confidence_scores.extend(m.confidence_scores)

        avg_confidence = np.mean(all_confidence_scores) if all_confidence_scores else 0.0
        std_confidence = np.std(all_confidence_scores) if all_confidence_scores else 0.0

        # Confidence interval
        ci = self._calculate_confidence_interval(
            avg_accuracy,
            base_metrics.total_questions
        )

        # Target validation
        in_target = self.target_range[0] <= avg_accuracy <= self.target_range[1]

        return ScenarioFinalResults(
            scenario_type=scenario_name,
            total_samples=base_metrics.total_questions,
            correct_answers=int(avg_accuracy * base_metrics.total_questions),
            accuracy=avg_accuracy,
            confidence_interval_95=ci,
            multi_run_stats=multi_run_stats,
            avg_processing_time=avg_processing_time,
            avg_retrieval_time=avg_retrieval_time,
            avg_generation_time=avg_generation_time,
            avg_confidence_score=avg_confidence,
            std_confidence_score=std_confidence,
            error_rate=base_metrics.error_rate,
            in_target_range=in_target,
            target_range=self.target_range
        )

    def _calculate_confidence_interval(
        self,
        accuracy: float,
        sample_count: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for accuracy.

        Uses Wilson score interval (more accurate for proportions).
        """
        if sample_count == 0:
            return (0.0, 0.0)

        # Z-score for 95% confidence
        z = 1.96

        p = accuracy
        n = sample_count

        # Wilson score interval
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower, upper)

    def _calculate_multi_run_statistics(
        self,
        accuracies: List[float],
        target_range: Tuple[float, float]
    ) -> MultiRunStatistics:
        """Calculate comprehensive multi-run statistics."""
        if not accuracies:
            return MultiRunStatistics(
                mean_accuracy=0.0,
                std_accuracy=0.0,
                median_accuracy=0.0,
                min_accuracy=0.0,
                max_accuracy=0.0,
                confidence_interval_95=(0.0, 0.0),
                coefficient_of_variation=0.0,
                individual_run_accuracies=[],
                statistical_significance=False
            )

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
        median_acc = np.median(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)

        # Coefficient of variation
        cv = (std_acc / mean_acc) if mean_acc > 0 else 0.0

        # Confidence interval for the mean (t-distribution)
        if len(accuracies) > 1:
            if SCIPY_AVAILABLE:
                from scipy.stats import t
                df = len(accuracies) - 1
                t_crit = t.ppf(0.975, df)  # 95% CI
            else:
                # Fallback: use normal approximation
                t_crit = 1.96

            margin = t_crit * std_acc / np.sqrt(len(accuracies))
            ci_lower = max(0.0, mean_acc - margin)
            ci_upper = min(1.0, mean_acc + margin)
        else:
            ci_lower, ci_upper = mean_acc, mean_acc

        # Statistical significance test (one-sample t-test against target midpoint)
        target_midpoint = (target_range[0] + target_range[1]) / 2
        if len(accuracies) > 1 and SCIPY_AVAILABLE:
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(accuracies, target_midpoint)
            significant = p_value < 0.05
        else:
            p_value = None
            significant = abs(mean_acc - target_midpoint) / std_acc > 1.96 if std_acc > 0 else False

        return MultiRunStatistics(
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            median_accuracy=median_acc,
            min_accuracy=min_acc,
            max_accuracy=max_acc,
            confidence_interval_95=(ci_lower, ci_upper),
            coefficient_of_variation=cv,
            individual_run_accuracies=accuracies,
            statistical_significance=significant,
            p_value=p_value
        )

    def _calculate_consistency(self, values: List[float]) -> float:
        """
        Calculate consistency metric (1 - coefficient of variation).

        Returns value between 0 and 1, where 1 is perfect consistency.
        """
        if not values or len(values) < 2:
            return 1.0

        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 1.0

        cv = std_val / mean_val
        consistency = max(0.0, 1.0 - cv)

        return consistency

    def _collect_performance_metrics(
        self,
        run_results: List[Dict[str, Any]]
    ) -> SystemPerformanceMetrics:
        """Collect comprehensive system performance metrics."""
        # Aggregate all processing times
        all_processing_times = []
        total_evaluation_time = 0.0
        peak_memory = 0.0
        avg_memory_list = []
        successful = 0
        total = 0

        for run in run_results:
            all_processing_times.extend(run["processing_times"])
            total_evaluation_time += run["evaluation_time"]
            peak_memory = max(peak_memory, run["peak_memory_gb"])
            avg_memory_list.append(run["peak_memory_gb"])
            successful += run["total_correct"]
            total += run["total_questions"]

        # Calculate timing percentiles
        if all_processing_times:
            p50 = np.percentile(all_processing_times, 50)
            p95 = np.percentile(all_processing_times, 95)
            p99 = np.percentile(all_processing_times, 99)
            avg_time = np.mean(all_processing_times)
        else:
            p50 = p95 = p99 = avg_time = 0.0

        # Memory metrics
        avg_memory = np.mean(avg_memory_list) if avg_memory_list else 0.0
        memory_limit = self.config.performance.memory_limit_gb
        memory_utilization = (peak_memory / memory_limit * 100) if memory_limit > 0 else 0.0
        memory_within_limit = peak_memory <= memory_limit

        # Reliability metrics
        success_rate = (successful / total) if total > 0 else 0.0
        error_rate = 1.0 - success_rate

        # Throughput metrics
        if total_evaluation_time > 0:
            qps = total / total_evaluation_time
            samples_per_min = (total / total_evaluation_time) * 60
        else:
            qps = samples_per_min = 0.0

        return SystemPerformanceMetrics(
            avg_query_time=avg_time,
            p50_query_time=p50,
            p95_query_time=p95,
            p99_query_time=p99,
            total_evaluation_time=total_evaluation_time,
            peak_memory_gb=peak_memory,
            avg_memory_gb=avg_memory,
            memory_utilization_percent=memory_utilization,
            memory_within_limit=memory_within_limit,
            total_queries=total,
            successful_queries=successful,
            failed_queries=total - successful,
            success_rate=success_rate,
            error_rate=error_rate,
            queries_per_second=qps,
            samples_per_minute=samples_per_min
        )

    def _assess_statistical_confidence(
        self,
        results: FinalValidationResults
    ) -> str:
        """Assess overall statistical confidence level."""
        confidence_factors = []

        # Factor 1: Sample size
        if results.total_questions >= 500:
            confidence_factors.append(2)  # High confidence
        elif results.total_questions >= 200:
            confidence_factors.append(1)  # Medium confidence
        else:
            confidence_factors.append(0)  # Low confidence

        # Factor 2: Multi-run consistency
        if results.num_evaluation_runs > 1:
            if results.cross_run_consistency >= 0.95:
                confidence_factors.append(2)
            elif results.cross_run_consistency >= 0.85:
                confidence_factors.append(1)
            else:
                confidence_factors.append(0)
        else:
            confidence_factors.append(1)  # Neutral for single run

        # Factor 3: Confidence interval width
        ci_width = results.overall_confidence_interval[1] - results.overall_confidence_interval[0]
        if ci_width <= 0.05:  # ±2.5%
            confidence_factors.append(2)
        elif ci_width <= 0.10:  # ±5%
            confidence_factors.append(1)
        else:
            confidence_factors.append(0)

        # Factor 4: Scenario consistency
        if results.scenario_consistency >= 0.90:
            confidence_factors.append(2)
        elif results.scenario_consistency >= 0.75:
            confidence_factors.append(1)
        else:
            confidence_factors.append(0)

        # Average confidence score
        avg_confidence = np.mean(confidence_factors)

        if avg_confidence >= 1.5:
            return "high"
        elif avg_confidence >= 0.75:
            return "medium"
        else:
            return "low"

    def _assess_production_readiness(
        self,
        results: FinalValidationResults
    ) -> str:
        """Assess production readiness status."""
        issues = []

        # Check accuracy target
        if not results.target_achieved:
            gap = abs(results.overall_accuracy -
                     (results.target_range[0] + results.target_range[1]) / 2)
            if gap > 0.10:
                issues.append("major_accuracy_gap")
            else:
                issues.append("minor_accuracy_gap")

        # Check performance
        perf = results.performance_metrics
        if perf.avg_query_time > 30.0:
            issues.append("slow_performance")

        # Check memory
        if not perf.memory_within_limit:
            issues.append("memory_overflow")

        # Check reliability
        if perf.success_rate < 0.95:
            issues.append("low_reliability")

        # Determine readiness
        if not issues:
            return "ready"
        elif any(issue.startswith("major") for issue in issues):
            return "not_ready"
        else:
            return "needs_optimization"

    def _generate_final_recommendations(
        self,
        results: FinalValidationResults
    ) -> List[str]:
        """Generate comprehensive final recommendations."""
        recommendations = []

        # Accuracy recommendations
        if not results.target_achieved:
            if results.overall_accuracy < results.target_range[0]:
                gap = results.target_range[0] - results.overall_accuracy
                recommendations.append(
                    f"CRITICAL: Accuracy {results.overall_accuracy:.1%} is {gap:.1%} below "
                    f"target minimum ({results.target_range[0]:.1%}). "
                    f"Recommend: (1) Enhanced prompt engineering, (2) Increase retrieval top-k, "
                    f"(3) Lower generation temperature for more deterministic outputs."
                )
            else:
                gap = results.overall_accuracy - results.target_range[1]
                recommendations.append(
                    f"Accuracy {results.overall_accuracy:.1%} exceeds target maximum by {gap:.1%}. "
                    f"Consider optimizing for inference speed while maintaining accuracy."
                )
        else:
            recommendations.append(
                f"✓ Target accuracy achieved: {results.overall_accuracy:.1%} is within "
                f"target range {results.target_range[0]:.1%}-{results.target_range[1]:.1%}"
            )

        # Scenario-specific recommendations
        for scenario_name, scenario_result in results.scenario_results.items():
            if not scenario_result.in_target_range:
                recommendations.append(
                    f"Scenario '{scenario_name}': {scenario_result.accuracy:.1%} outside target. "
                    f"Apply scenario-specific prompt optimization and parameter tuning."
                )

        # Performance recommendations
        perf = results.performance_metrics
        if perf.avg_query_time > 30.0:
            recommendations.append(
                f"Performance: Avg query time {perf.avg_query_time:.1f}s exceeds 30s target. "
                f"Recommend: (1) Batch processing optimization, (2) Reduce generation max_length, "
                f"(3) Optimize retrieval FAISS index."
            )
        else:
            recommendations.append(
                f"✓ Performance within target: {perf.avg_query_time:.1f}s per query"
            )

        # Memory recommendations
        if not perf.memory_within_limit:
            recommendations.append(
                f"Memory: Peak usage {perf.peak_memory_gb:.1f}GB exceeds limit. "
                f"Recommend: (1) More aggressive quantization, (2) Smaller batch sizes, "
                f"(3) Sequential model loading."
            )
        else:
            recommendations.append(
                f"✓ Memory usage within limit: {perf.peak_memory_gb:.1f}GB / "
                f"{self.config.performance.memory_limit_gb:.1f}GB"
            )

        # Statistical confidence
        if results.statistical_confidence == "high":
            recommendations.append(
                "✓ High statistical confidence in results - ready for publication/deployment"
            )
        elif results.statistical_confidence == "medium":
            recommendations.append(
                "Medium statistical confidence - consider additional evaluation runs for "
                "higher confidence in production deployment"
            )
        else:
            recommendations.append(
                "Low statistical confidence - recommend: (1) Additional evaluation runs, "
                "(2) Larger sample sizes, (3) More consistent optimization"
            )

        # Production readiness
        if results.production_readiness == "ready":
            recommendations.append(
                "✓ PRODUCTION READY: System meets all requirements for deployment"
            )
        elif results.production_readiness == "needs_optimization":
            recommendations.append(
                "Needs optimization before production: Address minor issues above"
            )
        else:
            recommendations.append(
                "NOT READY for production: Critical issues must be resolved"
            )

        return recommendations

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for reporting."""
        return {
            "model": {
                "vlm": self.config.model.vlm_name,
                "retriever": self.config.model.retriever_name,
                "quantization": self.config.model.quantization
            },
            "retrieval": {
                "top_k": self.config.retrieval.top_k,
                "embedding_dim": self.config.retrieval.embedding_dim
            },
            "generation": {
                "max_length": self.config.generation.max_length,
                "temperature": self.config.generation.temperature,
                "top_p": self.config.generation.top_p
            },
            "performance": {
                "memory_limit_gb": self.config.performance.memory_limit_gb,
                "batch_size": self.config.dataset.batch_size
            }
        }

    def _save_final_results(self, results: FinalValidationResults) -> None:
        """Save final validation results to disk."""
        try:
            # Save detailed JSON results
            results_file = self.output_dir / "sprint10_final_validation_results.json"
            with open(results_file, 'w') as f:
                json.dump(asdict(results), f, indent=2, default=str)

            # Save summary report
            summary_file = self.output_dir / "sprint10_summary_report.md"
            with open(summary_file, 'w') as f:
                f.write(self._generate_summary_report(results))

            logger.info(f"Final validation results saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def _generate_summary_report(self, results: FinalValidationResults) -> str:
        """Generate human-readable summary report."""
        report = f"""# Sprint 10: Final Accuracy Validation Report

## Executive Summary

**Overall Accuracy:** {results.overall_accuracy:.1%} (95% CI: [{results.overall_confidence_interval[0]:.1%}, {results.overall_confidence_interval[1]:.1%}])
**Target Range:** {results.target_range[0]:.1%} - {results.target_range[1]:.1%}
**Target Achieved:** {'✓ YES' if results.target_achieved else '✗ NO'}
**Total Questions:** {results.total_questions}
**Total Correct:** {results.total_correct}

**Evaluation Runs:** {results.num_evaluation_runs}
**Statistical Confidence:** {results.statistical_confidence.upper()}
**Production Readiness:** {results.production_readiness.upper()}

---

## Scenario Performance

| Scenario | Accuracy | 95% CI | Samples | In Target | Status |
|----------|----------|--------|---------|-----------|--------|
"""

        for scenario_name, scenario_result in results.scenario_results.items():
            status = "✓" if scenario_result.in_target_range else "✗"
            report += (
                f"| {scenario_name.upper():10} | {scenario_result.accuracy:6.1%} | "
                f"[{scenario_result.confidence_interval_95[0]:.1%}, {scenario_result.confidence_interval_95[1]:.1%}] | "
                f"{scenario_result.total_samples:7} | {scenario_result.in_target_range!s:9} | {status:6} |\n"
            )

        report += f"""
**Scenarios in Target:** {results.scenarios_in_target}/4
**Scenario Consistency:** {results.scenario_consistency:.1%}

---

## Performance Metrics

**Timing:**
- Average Query Time: {results.performance_metrics.avg_query_time:.2f}s
- P50 Query Time: {results.performance_metrics.p50_query_time:.2f}s
- P95 Query Time: {results.performance_metrics.p95_query_time:.2f}s
- P99 Query Time: {results.performance_metrics.p99_query_time:.2f}s
- Total Evaluation Time: {results.performance_metrics.total_evaluation_time:.1f}s

**Memory:**
- Peak Memory: {results.performance_metrics.peak_memory_gb:.2f}GB
- Average Memory: {results.performance_metrics.avg_memory_gb:.2f}GB
- Memory Utilization: {results.performance_metrics.memory_utilization_percent:.1f}%
- Within Limit: {'✓ YES' if results.performance_metrics.memory_within_limit else '✗ NO'}

**Reliability:**
- Success Rate: {results.performance_metrics.success_rate:.1%}
- Error Rate: {results.performance_metrics.error_rate:.1%}
- Queries Per Second: {results.performance_metrics.queries_per_second:.2f}

---

## Statistical Validation

"""

        if results.multi_run_statistics:
            mrs = results.multi_run_statistics
            report += f"""**Multi-Run Statistics (n={results.num_evaluation_runs}):**
- Mean Accuracy: {mrs.mean_accuracy:.1%} ± {mrs.std_accuracy:.1%}
- Median Accuracy: {mrs.median_accuracy:.1%}
- Range: [{mrs.min_accuracy:.1%}, {mrs.max_accuracy:.1%}]
- 95% CI of Mean: [{mrs.confidence_interval_95[0]:.1%}, {mrs.confidence_interval_95[1]:.1%}]
- Coefficient of Variation: {mrs.coefficient_of_variation:.1%}
- Cross-Run Consistency: {results.cross_run_consistency:.1%}
"""
            if mrs.p_value is not None:
                report += f"- Statistical Significance: {'YES' if mrs.statistical_significance else 'NO'} (p={mrs.p_value:.4f})\n"
        else:
            report += "Single evaluation run - no multi-run statistics available.\n"

        report += f"""
---

## Recommendations

"""
        for i, rec in enumerate(results.recommendations, 1):
            report += f"{i}. {rec}\n\n"

        report += f"""
---

## Configuration Summary

**Model Configuration:**
- VLM: {results.configuration_summary['model']['vlm']}
- Retriever: {results.configuration_summary['model']['retriever']}
- Quantization: {results.configuration_summary['model']['quantization']}

**Retrieval Configuration:**
- Top-K: {results.configuration_summary['retrieval']['top_k']}
- Embedding Dimension: {results.configuration_summary['retrieval']['embedding_dim']}

**Generation Configuration:**
- Max Length: {results.configuration_summary['generation']['max_length']}
- Temperature: {results.configuration_summary['generation']['temperature']}
- Top-P: {results.configuration_summary['generation']['top_p']}

**Performance Configuration:**
- Memory Limit: {results.configuration_summary['performance']['memory_limit_gb']}GB
- Batch Size: {results.configuration_summary['performance']['batch_size']}

---

**Report Generated:** {results.timestamp}
**Total Validation Time:** {results.total_validation_time:.1f}s
"""

        return report

    def _log_validation_summary(self, results: FinalValidationResults) -> None:
        """Log validation summary to console."""
        logger.info("\n" + "="*70)
        logger.info("SPRINT 10: FINAL ACCURACY VALIDATION COMPLETE")
        logger.info("="*70)
        logger.info(
            f"Overall Accuracy: {results.overall_accuracy:.1%} "
            f"(95% CI: [{results.overall_confidence_interval[0]:.1%}, "
            f"{results.overall_confidence_interval[1]:.1%}])"
        )
        logger.info(
            f"Target Range: {results.target_range[0]:.1%} - {results.target_range[1]:.1%}"
        )
        logger.info(
            f"Target Achieved: {'✓ YES' if results.target_achieved else '✗ NO'}"
        )
        logger.info(
            f"Total Questions: {results.total_questions} "
            f"(Correct: {results.total_correct})"
        )
        logger.info(
            f"Scenarios in Target: {results.scenarios_in_target}/4"
        )
        logger.info(
            f"Statistical Confidence: {results.statistical_confidence.upper()}"
        )
        logger.info(
            f"Production Readiness: {results.production_readiness.upper()}"
        )
        logger.info("="*70 + "\n")

    def cleanup(self) -> None:
        """Clean up validator resources."""
        logger.info("Cleaning up final validator...")
        if self.evaluator:
            self.evaluator.cleanup()
        self.memory_manager.emergency_cleanup()
        logger.info("Final validator cleanup completed")
