"""
MRAG-Bench Evaluation Framework

Core evaluation system implementing MRAG-Bench methodology for perspective change scenarios.
Provides comprehensive accuracy calculation, performance metrics, and analysis capabilities.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from collections import defaultdict
import re

from .results import EvaluationResults
from ..pipeline import MRAGPipeline, PipelineResult
from ..dataset import MRAGDataset, Sample
from ..config import MRAGConfig
from ..utils.error_handling import handle_errors, MRAGError
from ..utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class PerspectiveChangeType(Enum):
    """Perspective change scenario types from MRAG-Bench."""
    ANGLE = "angle"
    PARTIAL = "partial"
    SCOPE = "scope"
    OCCLUSION = "occlusion"


@dataclass
class ScenarioMetrics:
    """Metrics for a specific perspective change scenario."""
    scenario_type: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_processing_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    confidence_scores: List[float]
    error_count: int
    error_rate: float


@dataclass
class EvaluationSession:
    """Complete evaluation session results."""
    session_id: str
    timestamp: str
    config_summary: Dict[str, Any]
    scenario_results: Dict[str, ScenarioMetrics]
    overall_accuracy: float
    total_questions: int
    total_correct: int
    avg_processing_time: float
    memory_stats: Dict[str, float]
    error_analysis: Dict[str, Any]


class MRAGBenchEvaluator:
    """
    MRAG-Bench evaluation framework implementing comprehensive evaluation methodology.

    Features:
    - Perspective change scenario evaluation (angle, partial, scope, occlusion)
    - Automated accuracy calculation matching MRAG-Bench methodology
    - Performance metrics collection and analysis
    - Memory usage monitoring and optimization
    - Detailed error analysis and failure case identification
    - Configurable evaluation parameters and thresholds
    """

    def __init__(self, config: MRAGConfig, output_dir: str = "evaluation_results"):
        """
        Initialize MRAG-Bench evaluator.

        Args:
            config: Complete MRAG system configuration
            output_dir: Directory for evaluation results and reports
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.dataset = None
        self.pipeline = None
        self.memory_manager = MemoryManager(
            memory_limit_gb=config.performance.memory_limit_gb,
            buffer_gb=config.performance.memory_buffer_gb
        )

        # Evaluation state
        self.current_session = None
        self.results_cache = {}

        # Performance tracking
        self.evaluation_stats = {
            "sessions_completed": 0,
            "total_questions_evaluated": 0,
            "avg_accuracy_across_sessions": 0.0,
            "best_accuracy": 0.0,
            "worst_accuracy": 100.0
        }

        logger.info(f"MRAG-Bench evaluator initialized with output directory: {self.output_dir}")

    @handle_errors
    def initialize_components(self) -> None:
        """Initialize dataset and pipeline components."""
        logger.info("Initializing evaluation components...")

        # Initialize dataset
        if self.dataset is None:
            self.dataset = MRAGDataset(
                data_path=self.config.dataset.data_path,
                batch_size=self.config.dataset.batch_size,
                image_size=self.config.dataset.image_size
            )

            # Validate dataset
            validation_results = self.dataset.validate_dataset()
            if validation_results["status"] == "error":
                raise MRAGError(f"Dataset validation failed: {validation_results['errors']}")

            logger.info(f"Dataset initialized with {validation_results['total_samples']} samples")

        # Initialize pipeline
        if self.pipeline is None:
            self.pipeline = MRAGPipeline(self.config)
            self.pipeline.initialize_dataset()

        logger.info("Evaluation components initialized successfully")

    @handle_errors
    def evaluate_scenario(
        self,
        scenario_type: PerspectiveChangeType,
        max_samples: Optional[int] = None,
        use_cache: bool = True
    ) -> ScenarioMetrics:
        """
        Evaluate single perspective change scenario.

        Args:
            scenario_type: Type of perspective change scenario
            max_samples: Maximum number of samples to evaluate (None for all)
            use_cache: Whether to use cached results if available

        Returns:
            Detailed metrics for the scenario
        """
        logger.info(f"Evaluating {scenario_type.value} perspective change scenario...")

        # Check cache
        cache_key = f"{scenario_type.value}_{max_samples}"
        if use_cache and cache_key in self.results_cache:
            logger.info(f"Using cached results for {scenario_type.value}")
            return self.results_cache[cache_key]

        # Get samples for scenario
        samples = self.dataset.get_samples_by_scenario(scenario_type.value)
        # if not samples:
        #     raise MRAGError(f"No samples found for scenario: {scenario_type.value}")

        if max_samples is not None:
            samples = samples[:max_samples]

        logger.info(f"Processing {len(samples)} {scenario_type.value} samples...")

        # Process samples
        start_time = time.time()
        processing_times = []
        retrieval_times = []
        generation_times = []
        confidence_scores = []
        results = []
        errors = []

        for i, sample in enumerate(samples):
            try:
                logger.debug(f"Processing sample {i+1}/{len(samples)}: {sample.question_id}")

                # Process through pipeline - pass query image for image-based retrieval (MRAG-Bench format)
                result = self.pipeline.process_query(
                    question=sample.question,
                    question_id=sample.question_id,
                    ground_truth=sample.ground_truth,
                    query_image_path=sample.image_path,  # Use query image for retrieval
                    use_sequential_loading=False
                )

                results.append(result)
                processing_times.append(result.total_time)
                retrieval_times.append(result.retrieval_time)
                generation_times.append(result.generation_time)
                confidence_scores.append(result.confidence_score)

                # Log progress every 10 samples
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(samples)} {scenario_type.value} samples")

            except Exception as e:
                logger.error(f"Error processing sample {sample.question_id}: {e}")
                errors.append({
                    "question_id": sample.question_id,
                    "error": str(e),
                    "question": sample.question[:100] + "..." if len(sample.question) > 100 else sample.question
                })
                continue

        total_time = time.time() - start_time
        logger.info(f"Completed {scenario_type.value} evaluation in {total_time:.2f}s")

        # Calculate accuracy
        correct_count = 0
        for result, sample in zip(results, samples):
            if self._is_answer_correct(result.generated_answer, sample.ground_truth):
                correct_count += 1

        # Create scenario metrics
        metrics = ScenarioMetrics(
            scenario_type=scenario_type.value,
            total_questions=len(samples),
            correct_answers=correct_count,
            accuracy=correct_count / len(samples) if samples else 0.0,
            avg_processing_time=np.mean(processing_times) if processing_times else 0.0,
            avg_retrieval_time=np.mean(retrieval_times) if retrieval_times else 0.0,
            avg_generation_time=np.mean(generation_times) if generation_times else 0.0,
            confidence_scores=confidence_scores,
            error_count=len(errors),
            error_rate=len(errors) / len(samples) if samples else 0.0
        )

        # Cache results
        self.results_cache[cache_key] = metrics

        logger.info(
            f"{scenario_type.value} scenario results: "
            f"Accuracy: {metrics.accuracy:.1%} ({correct_count}/{len(samples)}), "
            f"Avg time: {metrics.avg_processing_time:.2f}s"
        )

        return metrics

    @handle_errors
    def evaluate_all_scenarios(
        self,
        max_samples_per_scenario: Optional[int] = None,
        target_accuracy_range: Tuple[float, float] = (0.53, 0.59)
    ) -> EvaluationSession:
        """
        Evaluate all perspective change scenarios.

        Args:
            max_samples_per_scenario: Maximum samples per scenario (None for all)
            target_accuracy_range: Target accuracy range for comparison

        Returns:
            Complete evaluation session results
        """
        logger.info("Starting comprehensive MRAG-Bench evaluation...")

        session_start = time.time()
        session_id = f"mrag_eval_{int(session_start)}"

        # Initialize components
        self.initialize_components()

        # Evaluate each scenario
        scenario_results = {}
        total_questions = 0
        total_correct = 0
        all_processing_times = []

        for scenario_type in PerspectiveChangeType:
            try:
                metrics = self.evaluate_scenario(
                    scenario_type=scenario_type,
                    max_samples=max_samples_per_scenario,
                    use_cache=False  # Fresh evaluation for comprehensive assessment
                )

                scenario_results[scenario_type.value] = metrics
                total_questions += metrics.total_questions
                total_correct += metrics.correct_answers
                all_processing_times.extend([metrics.avg_processing_time] * metrics.total_questions)

                logger.info(f"✓ {scenario_type.value}: {metrics.accuracy:.1%} accuracy")

            except Exception as e:
                logger.error(f"Failed to evaluate {scenario_type.value} scenario: {e}")
                # Create empty metrics for failed scenario
                scenario_results[scenario_type.value] = ScenarioMetrics(
                    scenario_type=scenario_type.value,
                    total_questions=0,
                    correct_answers=0,
                    accuracy=0.0,
                    avg_processing_time=0.0,
                    avg_retrieval_time=0.0,
                    avg_generation_time=0.0,
                    confidence_scores=[],
                    error_count=1,
                    error_rate=1.0
                )

        # Calculate overall metrics
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        avg_processing_time = np.mean(all_processing_times) if all_processing_times else 0.0

        # Memory statistics
        memory_stats = self.memory_manager.monitor.get_current_stats().__dict__

        # Error analysis
        error_analysis = self._analyze_errors(scenario_results)

        # Create evaluation session
        session = EvaluationSession(
            session_id=session_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config_summary=self._get_config_summary(),
            scenario_results=scenario_results,
            overall_accuracy=overall_accuracy,
            total_questions=total_questions,
            total_correct=total_correct,
            avg_processing_time=avg_processing_time,
            memory_stats=memory_stats,
            error_analysis=error_analysis
        )

        self.current_session = session

        # Update global statistics
        self._update_evaluation_stats(session)

        # Save results
        self._save_evaluation_session(session)

        total_time = time.time() - session_start
        logger.info(
            f"Comprehensive evaluation completed in {total_time:.1f}s:\n"
            f"  Overall Accuracy: {overall_accuracy:.1%} ({total_correct}/{total_questions})\n"
            f"  Target Range: {target_accuracy_range[0]:.1%} - {target_accuracy_range[1]:.1%}\n"
            f"  Target Achievement: {'✓' if target_accuracy_range[0] <= overall_accuracy <= target_accuracy_range[1] else '✗'}\n"
            f"  Avg Processing Time: {avg_processing_time:.2f}s per query"
        )

        return session

    def _is_answer_correct(self, generated_answer: str, ground_truth: str) -> bool:
        """
        Determine if generated answer matches ground truth using MRAG-Bench methodology.

        Args:
            generated_answer: Model generated answer
            ground_truth: Ground truth answer

        Returns:
            True if answer is considered correct
        """
        if not generated_answer or not ground_truth:
            return False

        # Normalize answers for comparison
        generated_norm = self._normalize_answer(generated_answer)
        ground_truth_norm = self._normalize_answer(ground_truth)

        # Exact match
        if generated_norm == ground_truth_norm:
            return True

        # Partial match with high overlap
        generated_words = set(generated_norm.split())
        ground_truth_words = set(ground_truth_norm.split())

        if not generated_words or not ground_truth_words:
            return False

        # Calculate overlap
        intersection = generated_words.intersection(ground_truth_words)
        union = generated_words.union(ground_truth_words)
        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        # Threshold for partial correctness
        return jaccard_similarity >= 0.6

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for consistent comparison."""
        # Convert to lowercase
        answer = answer.lower().strip()

        # Remove punctuation and extra whitespace
        answer = re.sub(r'[^\w\s]', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer)

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        words = [word for word in answer.split() if word not in stop_words]

        return ' '.join(words)

    def _analyze_errors(self, scenario_results: Dict[str, ScenarioMetrics]) -> Dict[str, Any]:
        """Analyze errors across scenarios to identify patterns."""
        error_analysis = {
            "total_errors": sum(metrics.error_count for metrics in scenario_results.values()),
            "error_by_scenario": {},
            "common_failure_patterns": [],
            "performance_bottlenecks": []
        }

        for scenario, metrics in scenario_results.items():
            error_analysis["error_by_scenario"][scenario] = {
                "error_count": metrics.error_count,
                "error_rate": metrics.error_rate,
                "accuracy": metrics.accuracy
            }

            # Identify performance bottlenecks
            if metrics.avg_processing_time > 30.0:  # Target: <30s per query
                error_analysis["performance_bottlenecks"].append(f"{scenario}: slow processing ({metrics.avg_processing_time:.1f}s)")

            if metrics.accuracy < 0.4:  # Significantly below target
                error_analysis["common_failure_patterns"].append(f"{scenario}: low accuracy ({metrics.accuracy:.1%})")

        return error_analysis

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration for evaluation records."""
        return {
            "model_names": {
                "vlm": self.config.model.vlm_name,
                "retriever": self.config.model.retriever_name
            },
            "quantization": self.config.model.quantization,
            "memory_limit": self.config.performance.memory_limit_gb,
            "retrieval_config": {
                "top_k": self.config.retrieval.top_k,
                "embedding_dim": self.config.retrieval.embedding_dim
            },
            "generation_config": {
                "max_length": self.config.generation.max_length,
                "temperature": self.config.generation.temperature
            }
        }

    def _update_evaluation_stats(self, session: EvaluationSession) -> None:
        """Update global evaluation statistics."""
        self.evaluation_stats["sessions_completed"] += 1
        self.evaluation_stats["total_questions_evaluated"] += session.total_questions

        # Update accuracy statistics
        current_avg = self.evaluation_stats["avg_accuracy_across_sessions"]
        n = self.evaluation_stats["sessions_completed"]
        new_avg = ((current_avg * (n-1)) + session.overall_accuracy) / n
        self.evaluation_stats["avg_accuracy_across_sessions"] = new_avg

        self.evaluation_stats["best_accuracy"] = max(
            self.evaluation_stats["best_accuracy"],
            session.overall_accuracy
        )
        self.evaluation_stats["worst_accuracy"] = min(
            self.evaluation_stats["worst_accuracy"],
            session.overall_accuracy
        )

    def _save_evaluation_session(self, session: EvaluationSession) -> None:
        """Save evaluation session results to disk."""
        try:
            # Save detailed results as JSON
            session_file = self.output_dir / f"{session.session_id}_detailed.json"
            with open(session_file, 'w') as f:
                json.dump(asdict(session), f, indent=2, default=str)

            # Save summary report
            summary_file = self.output_dir / f"{session.session_id}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(self._generate_session_summary(session))

            logger.info(f"Evaluation results saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save evaluation session: {e}")

    def _generate_session_summary(self, session: EvaluationSession) -> str:
        """Generate human-readable summary of evaluation session."""
        summary = f"""
MRAG-Bench Evaluation Session Summary
=====================================

Session ID: {session.session_id}
Timestamp: {session.timestamp}
Overall Accuracy: {session.overall_accuracy:.1%} ({session.total_correct}/{session.total_questions})
Average Processing Time: {session.avg_processing_time:.2f}s per query

Scenario Results:
-----------------
"""

        for scenario, metrics in session.scenario_results.items():
            summary += f"""
{scenario.upper()} Perspective Change:
  - Accuracy: {metrics.accuracy:.1%} ({metrics.correct_answers}/{metrics.total_questions})
  - Avg Processing Time: {metrics.avg_processing_time:.2f}s
  - Avg Retrieval Time: {metrics.avg_retrieval_time:.2f}s
  - Avg Generation Time: {metrics.avg_generation_time:.2f}s
  - Error Rate: {metrics.error_rate:.1%}
  - Avg Confidence: {np.mean(metrics.confidence_scores):.2f} (±{np.std(metrics.confidence_scores):.2f})
"""

        summary += f"""
Memory Usage:
-------------
  - GPU Allocated: {session.memory_stats.get('gpu_allocated_gb', 0):.2f}GB
  - GPU Cached: {session.memory_stats.get('gpu_cached_gb', 0):.2f}GB
  - System RAM: {session.memory_stats.get('system_ram_gb', 0):.2f}GB

Error Analysis:
---------------
  - Total Errors: {session.error_analysis['total_errors']}
  - Performance Bottlenecks: {len(session.error_analysis['performance_bottlenecks'])}
  - Failure Patterns: {len(session.error_analysis['common_failure_patterns'])}

Configuration:
--------------
  - VLM Model: {session.config_summary['model_names']['vlm']}
  - Retriever Model: {session.config_summary['model_names']['retriever']}
  - Quantization: {session.config_summary['quantization']}
  - Memory Limit: {session.config_summary['memory_limit']}GB
  - Top-K Retrieval: {session.config_summary['retrieval_config']['top_k']}
"""
        return summary

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        return {
            **self.evaluation_stats,
            "current_session": asdict(self.current_session) if self.current_session else None,
            "memory_stats": self.memory_manager.monitor.get_current_stats().__dict__
        }

    def cleanup(self) -> None:
        """Clean up evaluation resources."""
        logger.info("Cleaning up MRAG-Bench evaluator...")

        if self.pipeline:
            self.pipeline.cleanup()

        self.memory_manager.emergency_cleanup()

        logger.info("Evaluator cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()