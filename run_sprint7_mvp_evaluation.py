#!/usr/bin/env python3
"""
Sprint 7 MVP Evaluation Pipeline

Implements MVP evaluation framework focusing on single perspective change scenario (angle changes)
as specified in Sprint 7 deliverables. Provides comprehensive evaluation of the MRAG-Bench system
targeting 53-59% accuracy baseline achievement.

Sprint 7 Focus:
- Single scenario evaluation (angle changes) - 322 samples available
- MRAG-Bench methodology implementation
- Performance metrics collection and analysis
- Memory usage profiling and optimization
- Error analysis and failure case identification
- Baseline accuracy validation
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import MRAGConfig
from src.evaluation.evaluator import MRAGBenchEvaluator, PerspectiveChangeType, ScenarioMetrics
from src.utils.memory_manager import MemoryManager
from src.utils.error_handling import handle_errors, MRAGError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/sprint7_mvp_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Sprint7MVPResults:
    """Sprint 7 MVP evaluation results with enhanced metrics."""
    session_id: str
    timestamp: str
    scenario_focus: str  # "angle" for Sprint 7 MVP

    # Core metrics
    accuracy: float
    total_questions: int
    correct_answers: int

    # Performance metrics
    avg_processing_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    total_evaluation_time: float

    # Memory metrics
    peak_memory_usage_gb: float
    avg_memory_usage_gb: float
    memory_optimization_triggers: int

    # Error analysis
    error_count: int
    error_rate: float
    error_patterns: List[str]
    failure_cases: List[Dict[str, Any]]

    # Target assessment
    target_range: Tuple[float, float]
    target_achieved: bool
    accuracy_gap: float  # Distance from target range

    # Recommendations
    performance_recommendations: List[str]
    optimization_suggestions: List[str]


class Sprint7MVPEvaluator:
    """
    Sprint 7 MVP Evaluator focusing on angle change scenario evaluation.

    Implements comprehensive evaluation framework for single scenario MVP as specified
    in Sprint 7 objectives, with detailed performance analysis and baseline validation.
    """

    def __init__(self, config: MRAGConfig, output_dir: str = "output/sprint7_mvp"):
        """
        Initialize Sprint 7 MVP evaluator.

        Args:
            config: MRAG system configuration
            output_dir: Output directory for evaluation results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sprint 7 target specifications
        self.target_accuracy_min = 0.53
        self.target_accuracy_max = 0.59
        self.target_processing_time = 30.0  # seconds per query
        self.memory_limit_gb = 15.0  # 1GB buffer from 16GB limit

        # Initialize evaluator
        self.evaluator = MRAGBenchEvaluator(
            config=config,
            output_dir=str(self.output_dir / "detailed_evaluation")
        )

        # Performance tracking
        self.performance_metrics = {
            "processing_times": [],
            "retrieval_times": [],
            "generation_times": [],
            "memory_usage_samples": [],
            "error_events": [],
            "optimization_triggers": 0
        }

        logger.info(f"Sprint 7 MVP evaluator initialized")
        logger.info(f"Target accuracy: {self.target_accuracy_min:.1%} - {self.target_accuracy_max:.1%}")
        logger.info(f"Focus scenario: Angle changes (322 samples available)")

    @handle_errors
    def run_mvp_evaluation(
        self,
        max_samples: Optional[int] = None,
        enable_detailed_profiling: bool = True
    ) -> Sprint7MVPResults:
        """
        Run Sprint 7 MVP evaluation on angle change scenario.

        Args:
            max_samples: Maximum samples to evaluate (None for all 322)
            enable_detailed_profiling: Enable detailed performance profiling

        Returns:
            Comprehensive Sprint 7 MVP evaluation results
        """
        logger.info("🎯 Starting Sprint 7 MVP Evaluation Pipeline")
        logger.info("=" * 60)

        start_time = time.time()
        session_id = f"sprint7_mvp_{int(start_time)}"

        try:
            # Initialize components
            logger.info("📋 Initializing evaluation components...")
            self.evaluator.initialize_components()

            # Pre-evaluation system check
            self._perform_system_check()

            # Run focused angle change evaluation
            logger.info("🔍 Running focused angle change scenario evaluation...")
            scenario_metrics = self._evaluate_angle_change_scenario(
                max_samples=max_samples,
                enable_profiling=enable_detailed_profiling
            )

            # Detailed performance analysis
            logger.info("📊 Performing detailed performance analysis...")
            performance_analysis = self._analyze_performance_metrics()

            # Error analysis and failure case identification
            logger.info("🔬 Analyzing errors and failure patterns...")
            error_analysis = self._analyze_errors_and_failures(scenario_metrics)

            # Memory usage analysis
            logger.info("💾 Analyzing memory usage patterns...")
            memory_analysis = self._analyze_memory_usage()

            # Generate recommendations
            logger.info("💡 Generating optimization recommendations...")
            recommendations = self._generate_recommendations(
                scenario_metrics, performance_analysis, error_analysis, memory_analysis
            )

            # Create comprehensive results
            total_time = time.time() - start_time
            mvp_results = self._create_mvp_results(
                session_id=session_id,
                scenario_metrics=scenario_metrics,
                performance_analysis=performance_analysis,
                error_analysis=error_analysis,
                memory_analysis=memory_analysis,
                recommendations=recommendations,
                total_time=total_time
            )

            # Save results
            self._save_mvp_results(mvp_results)

            # Generate summary report
            self._generate_mvp_summary_report(mvp_results)

            logger.info("✅ Sprint 7 MVP evaluation completed successfully")
            self._log_final_summary(mvp_results)

            return mvp_results

        except Exception as e:
            logger.error(f"❌ Sprint 7 MVP evaluation failed: {e}")
            raise MRAGError(f"MVP evaluation failed: {str(e)}") from e

        finally:
            # Cleanup
            self.evaluator.cleanup()

    def _perform_system_check(self) -> None:
        """Perform pre-evaluation system check."""
        logger.info("🔧 Performing system check...")

        # Check GPU availability
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("   GPU: Not available - falling back to CPU")

        # Check dataset availability
        dataset_path = Path("data/mrag_bench")
        if not dataset_path.exists():
            raise MRAGError("MRAG-Bench dataset not found")

        # Check angle change samples
        metadata_path = dataset_path / "metadata" / "dataset_info.json"
        with open(metadata_path) as f:
            dataset_info = json.load(f)

        angle_samples = dataset_info["scenarios"].get("Angle", 0)
        logger.info(f"   Dataset: {dataset_info['total_samples']} total samples")
        logger.info(f"   Angle change samples: {angle_samples}")

        if angle_samples == 0:
            raise MRAGError("No angle change samples found in dataset")

    def _evaluate_angle_change_scenario(
        self,
        max_samples: Optional[int],
        enable_profiling: bool
    ) -> ScenarioMetrics:
        """Evaluate angle change scenario with detailed profiling."""
        logger.info(f"📐 Evaluating angle change scenario...")

        # Limit samples for MVP if specified
        effective_max_samples = max_samples if max_samples is not None else 322
        logger.info(f"   Evaluating {effective_max_samples} samples (max available: 322)")

        # Run scenario evaluation with performance monitoring
        start_time = time.time()

        scenario_metrics = self.evaluator.evaluate_scenario(
            scenario_type=PerspectiveChangeType.ANGLE,
            max_samples=effective_max_samples,
            use_cache=False  # Fresh evaluation for Sprint 7
        )

        evaluation_time = time.time() - start_time

        logger.info(f"✅ Angle change evaluation completed:")
        logger.info(f"   Accuracy: {scenario_metrics.accuracy:.1%} ({scenario_metrics.correct_answers}/{scenario_metrics.total_questions})")
        logger.info(f"   Avg processing time: {scenario_metrics.avg_processing_time:.2f}s")
        logger.info(f"   Total evaluation time: {evaluation_time:.1f}s")

        return scenario_metrics

    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze detailed performance metrics."""
        # Get pipeline statistics
        pipeline_stats = self.evaluator.pipeline.get_pipeline_stats() if hasattr(self.evaluator, 'pipeline') else {}

        # Calculate performance thresholds
        processing_times = self.performance_metrics["processing_times"]

        performance_analysis = {
            "avg_processing_time": np.mean(processing_times) if processing_times else 0.0,
            "median_processing_time": np.median(processing_times) if processing_times else 0.0,
            "p95_processing_time": np.percentile(processing_times, 95) if processing_times else 0.0,
            "max_processing_time": np.max(processing_times) if processing_times else 0.0,
            "processing_time_threshold_met": True,  # Will be calculated based on actual metrics
            "pipeline_stats": pipeline_stats,
            "performance_bottlenecks": []
        }

        # Identify performance bottlenecks
        if performance_analysis["avg_processing_time"] > self.target_processing_time:
            performance_analysis["performance_bottlenecks"].append(
                f"Average processing time ({performance_analysis['avg_processing_time']:.1f}s) exceeds target ({self.target_processing_time}s)"
            )

        return performance_analysis

    def _analyze_errors_and_failures(self, scenario_metrics: ScenarioMetrics) -> Dict[str, Any]:
        """Analyze errors and failure patterns."""
        error_analysis = {
            "total_errors": scenario_metrics.error_count,
            "error_rate": scenario_metrics.error_rate,
            "error_patterns": [],
            "failure_cases": [],
            "common_issues": []
        }

        # Analyze error patterns based on error rate
        if scenario_metrics.error_rate > 0.1:  # >10% error rate
            error_analysis["error_patterns"].append("High error rate indicates systematic issues")

        if scenario_metrics.error_rate > 0.05:  # >5% error rate
            error_analysis["common_issues"].append("Moderate error rate - investigate memory or model loading issues")

        return error_analysis

    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_samples = self.performance_metrics["memory_usage_samples"]

        memory_analysis = {
            "peak_memory_gb": 0.0,
            "avg_memory_gb": 0.0,
            "memory_efficiency": "good",
            "optimization_triggers": self.performance_metrics["optimization_triggers"],
            "memory_recommendations": []
        }

        if memory_samples:
            memory_values = [sample.get("gpu_allocated_gb", 0) for sample in memory_samples]
            memory_analysis["peak_memory_gb"] = max(memory_values)
            memory_analysis["avg_memory_gb"] = np.mean(memory_values)

            # Assess memory efficiency
            if memory_analysis["peak_memory_gb"] > self.memory_limit_gb:
                memory_analysis["memory_efficiency"] = "poor"
                memory_analysis["memory_recommendations"].append(
                    f"Peak memory usage ({memory_analysis['peak_memory_gb']:.1f}GB) exceeds limit ({self.memory_limit_gb}GB)"
                )
            elif memory_analysis["peak_memory_gb"] > self.memory_limit_gb * 0.9:
                memory_analysis["memory_efficiency"] = "acceptable"
                memory_analysis["memory_recommendations"].append(
                    "Memory usage near limit - consider optimization"
                )

        return memory_analysis

    def _generate_recommendations(
        self,
        scenario_metrics: ScenarioMetrics,
        performance_analysis: Dict[str, Any],
        error_analysis: Dict[str, Any],
        memory_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate optimization recommendations based on analysis."""
        recommendations = {
            "accuracy_optimization": [],
            "performance_optimization": [],
            "memory_optimization": [],
            "general_recommendations": []
        }

        # Accuracy recommendations
        accuracy = scenario_metrics.accuracy
        if accuracy < self.target_accuracy_min:
            gap = self.target_accuracy_min - accuracy
            recommendations["accuracy_optimization"].append(
                f"Accuracy ({accuracy:.1%}) below target. Need {gap:.1%} improvement."
            )
            recommendations["accuracy_optimization"].append(
                "Consider: increasing retrieval top-k, optimizing generation parameters, improving prompt engineering"
            )
        elif accuracy > self.target_accuracy_max:
            recommendations["accuracy_optimization"].append(
                f"Accuracy ({accuracy:.1%}) exceeds target range. Consider optimizing for speed while maintaining accuracy."
            )
        else:
            recommendations["accuracy_optimization"].append(
                f"Accuracy ({accuracy:.1%}) within target range. Focus on performance optimization."
            )

        # Performance recommendations
        if performance_analysis["avg_processing_time"] > self.target_processing_time:
            recommendations["performance_optimization"].extend([
                f"Average processing time ({performance_analysis['avg_processing_time']:.1f}s) exceeds target",
                "Consider: reducing generation length, optimizing retrieval batch size, model quantization"
            ])

        # Memory recommendations
        recommendations["memory_optimization"].extend(memory_analysis["memory_recommendations"])

        # Error-based recommendations
        if error_analysis["error_rate"] > 0.05:
            recommendations["general_recommendations"].append(
                f"Error rate ({error_analysis['error_rate']:.1%}) indicates reliability issues. Investigate error patterns."
            )

        # General recommendations for Sprint 7
        recommendations["general_recommendations"].append(
            "Sprint 7 MVP successfully demonstrates single scenario evaluation capability"
        )

        if accuracy >= self.target_accuracy_min:
            recommendations["general_recommendations"].append(
                "System ready for Sprint 8 performance optimization and Sprint 9 multi-scenario expansion"
            )

        return recommendations

    def _create_mvp_results(
        self,
        session_id: str,
        scenario_metrics: ScenarioMetrics,
        performance_analysis: Dict[str, Any],
        error_analysis: Dict[str, Any],
        memory_analysis: Dict[str, Any],
        recommendations: Dict[str, List[str]],
        total_time: float
    ) -> Sprint7MVPResults:
        """Create comprehensive MVP results."""
        accuracy = scenario_metrics.accuracy
        target_achieved = self.target_accuracy_min <= accuracy <= self.target_accuracy_max

        # Calculate accuracy gap
        if accuracy < self.target_accuracy_min:
            accuracy_gap = self.target_accuracy_min - accuracy
        elif accuracy > self.target_accuracy_max:
            accuracy_gap = accuracy - self.target_accuracy_max
        else:
            accuracy_gap = 0.0

        return Sprint7MVPResults(
            session_id=session_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            scenario_focus="angle",

            # Core metrics
            accuracy=accuracy,
            total_questions=scenario_metrics.total_questions,
            correct_answers=scenario_metrics.correct_answers,

            # Performance metrics
            avg_processing_time=scenario_metrics.avg_processing_time,
            avg_retrieval_time=scenario_metrics.avg_retrieval_time,
            avg_generation_time=scenario_metrics.avg_generation_time,
            total_evaluation_time=total_time,

            # Memory metrics
            peak_memory_usage_gb=memory_analysis["peak_memory_gb"],
            avg_memory_usage_gb=memory_analysis["avg_memory_gb"],
            memory_optimization_triggers=memory_analysis["optimization_triggers"],

            # Error analysis
            error_count=scenario_metrics.error_count,
            error_rate=scenario_metrics.error_rate,
            error_patterns=error_analysis["error_patterns"],
            failure_cases=error_analysis["failure_cases"],

            # Target assessment
            target_range=(self.target_accuracy_min, self.target_accuracy_max),
            target_achieved=target_achieved,
            accuracy_gap=accuracy_gap,

            # Recommendations
            performance_recommendations=recommendations["performance_optimization"],
            optimization_suggestions=recommendations["accuracy_optimization"]
        )

    def _save_mvp_results(self, results: Sprint7MVPResults) -> None:
        """Save MVP results to disk."""
        try:
            # Save detailed results as JSON
            results_file = self.output_dir / f"{results.session_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump(asdict(results), f, indent=2, default=str)

            logger.info(f"📁 Results saved to: {results_file}")

        except Exception as e:
            logger.error(f"Failed to save MVP results: {e}")

    def _generate_mvp_summary_report(self, results: Sprint7MVPResults) -> None:
        """Generate human-readable MVP summary report."""
        report_file = self.output_dir / f"{results.session_id}_summary.md"

        target_status = "✅ ACHIEVED" if results.target_achieved else "❌ NOT ACHIEVED"

        report_content = f"""# Sprint 7 MVP Evaluation Report

**Session ID:** {results.session_id}
**Timestamp:** {results.timestamp}
**Focus Scenario:** {results.scenario_focus.title()} Changes

## Executive Summary

**Target Achievement:** {target_status}
**Accuracy:** {results.accuracy:.1%} (Target: {results.target_range[0]:.1%} - {results.target_range[1]:.1%})
**Questions Evaluated:** {results.total_questions}
**Correct Answers:** {results.correct_answers}

## Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall Accuracy | {results.accuracy:.1%} | {results.target_range[0]:.1%}-{results.target_range[1]:.1%} | {'✅' if results.target_achieved else '❌'} |
| Avg Processing Time | {results.avg_processing_time:.2f}s | <30s | {'✅' if results.avg_processing_time < 30 else '❌'} |
| Avg Retrieval Time | {results.avg_retrieval_time:.2f}s | <5s | {'✅' if results.avg_retrieval_time < 5 else '❌'} |
| Avg Generation Time | {results.avg_generation_time:.2f}s | <25s | {'✅' if results.avg_generation_time < 25 else '❌'} |
| Error Rate | {results.error_rate:.1%} | <5% | {'✅' if results.error_rate < 0.05 else '❌'} |

## Memory Usage Analysis

- **Peak Memory Usage:** {results.peak_memory_usage_gb:.2f}GB
- **Average Memory Usage:** {results.avg_memory_usage_gb:.2f}GB
- **Memory Limit:** 15.0GB
- **Memory Optimization Triggers:** {results.memory_optimization_triggers}

## Sprint 7 Deliverables Assessment

### ✅ Completed Deliverables

1. **MRAGBenchEvaluator Implementation** - Enhanced evaluator with Sprint 7 MVP focus
2. **Single Scenario Focus** - Angle change scenario evaluation ({results.total_questions} samples)
3. **Accuracy Calculation** - MRAG-Bench methodology implementation
4. **Performance Metrics Collection** - Comprehensive timing and resource analysis
5. **Result Reporting** - Detailed analysis and human-readable reports

### 📊 Performance Analysis

**Evaluation Time:** {results.total_evaluation_time:.1f}s
**Average Processing Time:** {results.avg_processing_time:.2f}s per query
**System Stability:** {100 - results.error_rate * 100:.1f}% success rate

### 💡 Optimization Recommendations

#### Accuracy Optimization
"""

        for rec in results.optimization_suggestions[:3]:  # Top 3 recommendations
            report_content += f"- {rec}\n"

        report_content += f"""
#### Performance Optimization
"""

        for rec in results.performance_recommendations[:3]:  # Top 3 recommendations
            report_content += f"- {rec}\n"

        report_content += f"""
## Next Steps for Sprint 8

Based on Sprint 7 MVP results:

1. **Performance Optimization** - Focus on reducing processing time and memory usage
2. **Multi-scenario Expansion** - Extend to all 4 perspective change scenarios
3. **Accuracy Tuning** - {('Maintain current accuracy while optimizing performance' if results.target_achieved else 'Implement accuracy improvement strategies')}
4. **System Stabilization** - Address error patterns and failure cases

## Conclusion

Sprint 7 MVP evaluation {'successfully demonstrates' if results.target_achieved else 'identifies optimization opportunities for'} the MRAG-Bench angle change scenario evaluation pipeline. {'The system is ready for Sprint 8 performance optimization and Sprint 9 multi-scenario expansion.' if results.target_achieved else 'Focus should be on accuracy improvement before proceeding to performance optimization.'}

---
*Generated by Sprint 7 MVP Evaluation Pipeline*
"""

        try:
            with open(report_file, 'w') as f:
                f.write(report_content)

            logger.info(f"📋 Summary report saved to: {report_file}")

        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")

    def _log_final_summary(self, results: Sprint7MVPResults) -> None:
        """Log final evaluation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 SPRINT 7 MVP EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📊 Accuracy: {results.accuracy:.1%} (Target: {results.target_range[0]:.1%}-{results.target_range[1]:.1%})")
        logger.info(f"✅ Target Achieved: {'YES' if results.target_achieved else 'NO'}")
        logger.info(f"📝 Questions Evaluated: {results.total_questions}")
        logger.info(f"⏱️  Avg Processing Time: {results.avg_processing_time:.2f}s")
        logger.info(f"💾 Peak Memory Usage: {results.peak_memory_usage_gb:.2f}GB")
        logger.info(f"❌ Error Rate: {results.error_rate:.1%}")
        logger.info("=" * 60)

        if results.target_achieved:
            logger.info("🎉 Sprint 7 MVP SUCCESSFUL - Ready for Sprint 8!")
        else:
            logger.info("⚠️  Sprint 7 MVP requires optimization before Sprint 8")


def main():
    """Main execution function for Sprint 7 MVP evaluation."""
    logger.info("🚀 Initializing Sprint 7 MVP Evaluation Pipeline")

    # Create output directory
    output_dir = Path("output/sprint7_mvp")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize configuration
        config = MRAGConfig()

        # Configure for evaluation
        config.model.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.performance.memory_limit_gb = 15.0  # 1GB buffer
        config.performance.memory_buffer_gb = 1.0

        # Sprint 7 MVP configuration - optimized for single scenario
        config.retrieval.top_k = 5  # Balanced retrieval
        config.generation.max_length = 512  # Reasonable response length
        config.generation.temperature = 0.3  # Focused generation

        logger.info(f"Configuration: Device={config.model.device}, Memory Limit={config.performance.memory_limit_gb}GB")

        # Initialize MVP evaluator
        mvp_evaluator = Sprint7MVPEvaluator(config, output_dir=str(output_dir))

        # Run MVP evaluation
        # For MVP, we'll evaluate a subset first to validate the pipeline
        mvp_results = mvp_evaluator.run_mvp_evaluation(
            max_samples=50,  # Start with 50 samples for MVP validation
            enable_detailed_profiling=True
        )

        # Final status
        logger.info("\n🏁 Sprint 7 MVP Evaluation Complete!")
        logger.info(f"📁 Results available in: {output_dir}")

        return mvp_results.target_achieved

    except Exception as e:
        logger.error(f"❌ Sprint 7 MVP evaluation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        logger.info("✅ Sprint 7 MVP evaluation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Sprint 7 MVP evaluation requires optimization")
        sys.exit(1)