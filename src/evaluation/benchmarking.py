"""
Performance Benchmarking System for Sprint 8

Implements comprehensive performance measurement and analysis for MRAG-Bench system:
- Timing benchmarks for all pipeline components
- Memory usage profiling throughout execution
- Accuracy measurement across different scenarios
- Throughput and latency analysis
- Comparative benchmarking for optimization validation
"""

import time
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import torch

from .performance_optimizer import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark execution result."""
    benchmark_name: str
    execution_time: float
    memory_usage_gb: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the suite."""
        self.results.append(result)

    def finalize(self):
        """Mark benchmark suite as complete."""
        self.end_time = time.time()

    @property
    def total_time(self) -> float:
        """Get total execution time."""
        return self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time."""
        if not self.results:
            return 0.0
        return statistics.mean(r.execution_time for r in self.results)

    @property
    def avg_memory_usage(self) -> float:
        """Calculate average memory usage."""
        if not self.results:
            return 0.0
        return statistics.mean(r.memory_usage_gb for r in self.results)


class PerformanceBenchmark:
    """
    Performance benchmarking utility for timing and profiling.

    Provides context manager for timing operations and collecting memory usage.
    """

    def __init__(self, name: str):
        """
        Initialize performance benchmark.

        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.start_memory = 0.0
        self.end_memory = 0.0
        self.success = True
        self.error = None
        self.metadata = {}

    def __enter__(self):
        """Start benchmark timing and memory tracking."""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated() / 1e9
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End benchmark and record results."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_memory = torch.cuda.memory_allocated() / 1e9
        self.end_time = time.time()

        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)

        return False  # Don't suppress exceptions

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    @property
    def memory_used(self) -> float:
        """Get memory used in GB."""
        return self.end_memory - self.start_memory if self.end_memory > 0 else 0.0

    def to_result(self) -> BenchmarkResult:
        """Convert to BenchmarkResult."""
        return BenchmarkResult(
            benchmark_name=self.name,
            execution_time=self.elapsed_time,
            memory_usage_gb=self.memory_used,
            success=self.success,
            metadata=self.metadata.copy(),
            error=self.error
        )


class ComponentBenchmarks:
    """
    Benchmarking suite for individual pipeline components.

    Tests retrieval, generation, and end-to-end pipeline performance.
    """

    def __init__(self):
        """Initialize component benchmarks."""
        self.results = {}

    def benchmark_retrieval(
        self,
        retriever,
        test_queries: List[str],
        warmup: int = 2
    ) -> BenchmarkSuite:
        """
        Benchmark retrieval performance.

        Args:
            retriever: Retrieval pipeline instance
            test_queries: List of test queries
            warmup: Number of warmup iterations

        Returns:
            Benchmark suite with results
        """
        suite = BenchmarkSuite(suite_name="retrieval_benchmark")

        # Warmup
        logger.info(f"Running {warmup} warmup iterations for retrieval...")
        for i in range(warmup):
            if test_queries:
                try:
                    retriever.encode_query(test_queries[0])
                except:
                    pass

        # Actual benchmarks
        logger.info(f"Benchmarking retrieval on {len(test_queries)} queries...")
        for i, query in enumerate(test_queries):
            with PerformanceBenchmark(f"retrieval_query_{i}") as bench:
                try:
                    bench.metadata["query"] = query
                    results = retriever.retrieve(query, top_k=5)
                    bench.metadata["results_count"] = len(results)
                except Exception as e:
                    logger.error(f"Retrieval benchmark failed for query {i}: {e}")

            suite.add_result(bench.to_result())

        suite.finalize()
        self.results["retrieval"] = suite
        return suite

    def benchmark_generation(
        self,
        generator,
        test_contexts: List[Any],
        warmup: int = 1
    ) -> BenchmarkSuite:
        """
        Benchmark generation performance.

        Args:
            generator: Generation pipeline instance
            test_contexts: List of test multimodal contexts
            warmup: Number of warmup iterations

        Returns:
            Benchmark suite with results
        """
        suite = BenchmarkSuite(suite_name="generation_benchmark")

        # Warmup
        logger.info(f"Running {warmup} warmup iteration for generation...")
        if test_contexts and warmup > 0:
            try:
                generator.generate_answer(test_contexts[0])
            except:
                pass

        # Actual benchmarks
        logger.info(f"Benchmarking generation on {len(test_contexts)} contexts...")
        for i, context in enumerate(test_contexts):
            with PerformanceBenchmark(f"generation_context_{i}") as bench:
                try:
                    bench.metadata["question"] = getattr(context, "question", "unknown")
                    result = generator.generate_answer(context)
                    bench.metadata["answer_length"] = len(result.generated_text) if hasattr(result, "generated_text") else 0
                except Exception as e:
                    logger.error(f"Generation benchmark failed for context {i}: {e}")

            suite.add_result(bench.to_result())

        suite.finalize()
        self.results["generation"] = suite
        return suite

    def benchmark_end_to_end(
        self,
        pipeline,
        test_samples: List[Any],
        warmup: int = 1
    ) -> BenchmarkSuite:
        """
        Benchmark end-to-end pipeline performance.

        Args:
            pipeline: Complete MRAG pipeline instance
            test_samples: List of test samples
            warmup: Number of warmup iterations

        Returns:
            Benchmark suite with results
        """
        suite = BenchmarkSuite(suite_name="end_to_end_benchmark")

        # Warmup
        if test_samples and warmup > 0:
            logger.info(f"Running {warmup} warmup iteration for end-to-end pipeline...")
            try:
                sample = test_samples[0]
                pipeline.process_query(
                    question=getattr(sample, "question", "test"),
                    image_paths=getattr(sample, "image_paths", [])
                )
            except:
                pass

        # Actual benchmarks
        logger.info(f"Benchmarking end-to-end pipeline on {len(test_samples)} samples...")
        for i, sample in enumerate(test_samples):
            with PerformanceBenchmark(f"e2e_sample_{i}") as bench:
                try:
                    bench.metadata["question_id"] = getattr(sample, "question_id", f"sample_{i}")
                    result = pipeline.process_query(
                        question=getattr(sample, "question", ""),
                        image_paths=getattr(sample, "image_paths", [])
                    )
                    bench.metadata["retrieval_time"] = getattr(result, "retrieval_time", 0.0)
                    bench.metadata["generation_time"] = getattr(result, "generation_time", 0.0)
                except Exception as e:
                    logger.error(f"End-to-end benchmark failed for sample {i}: {e}")

            suite.add_result(bench.to_result())

        suite.finalize()
        self.results["end_to_end"] = suite
        return suite

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.

        Returns:
            Dictionary with benchmark analysis
        """
        report = {
            "timestamp": time.time(),
            "components": {},
            "summary": {}
        }

        total_tests = 0
        total_successes = 0
        all_times = []
        all_memory = []

        for component_name, suite in self.results.items():
            component_report = {
                "total_tests": len(suite.results),
                "successful_tests": sum(1 for r in suite.results if r.success),
                "success_rate": suite.success_rate,
                "total_time": suite.total_time,
                "avg_execution_time": suite.avg_execution_time,
                "median_execution_time": statistics.median(r.execution_time for r in suite.results) if suite.results else 0.0,
                "p95_execution_time": np.percentile([r.execution_time for r in suite.results], 95) if suite.results else 0.0,
                "p99_execution_time": np.percentile([r.execution_time for r in suite.results], 99) if suite.results else 0.0,
                "avg_memory_usage": suite.avg_memory_usage,
                "peak_memory_usage": max((r.memory_usage_gb for r in suite.results), default=0.0),
                "failed_tests": [r.benchmark_name for r in suite.results if not r.success]
            }

            report["components"][component_name] = component_report

            total_tests += len(suite.results)
            total_successes += sum(1 for r in suite.results if r.success)
            all_times.extend(r.execution_time for r in suite.results)
            all_memory.extend(r.memory_usage_gb for r in suite.results)

        # Overall summary
        report["summary"] = {
            "total_tests": total_tests,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_tests if total_tests > 0 else 0.0,
            "avg_execution_time": statistics.mean(all_times) if all_times else 0.0,
            "p95_execution_time": np.percentile(all_times, 95) if all_times else 0.0,
            "avg_memory_usage": statistics.mean(all_memory) if all_memory else 0.0,
            "peak_memory_usage": max(all_memory) if all_memory else 0.0
        }

        return report


class AccuracyBenchmarks:
    """
    Benchmarking suite for accuracy measurement.

    Tests accuracy across different scenarios and configurations.
    """

    def __init__(self):
        """Initialize accuracy benchmarks."""
        self.results = {}

    def benchmark_scenario_accuracy(
        self,
        evaluator,
        scenario_type: str,
        max_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark accuracy for a specific scenario.

        Args:
            evaluator: MRAGBenchEvaluator instance
            scenario_type: Type of perspective change scenario
            max_samples: Maximum number of samples to test

        Returns:
            Accuracy benchmark results
        """
        logger.info(f"Benchmarking accuracy for scenario: {scenario_type}")

        with PerformanceBenchmark(f"accuracy_{scenario_type}") as bench:
            try:
                # Use evaluator to test scenario
                # This would call actual evaluation in production
                from .results import PerspectiveChangeType

                scenario_enum = PerspectiveChangeType[scenario_type.upper()]
                metrics = evaluator.evaluate_scenario(
                    scenario_type=scenario_enum,
                    max_samples=max_samples
                )

                bench.metadata.update({
                    "scenario": scenario_type,
                    "total_questions": metrics.total_questions,
                    "correct_answers": metrics.correct_answers,
                    "accuracy": metrics.accuracy,
                    "avg_processing_time": metrics.avg_processing_time,
                    "error_rate": metrics.error_rate
                })

            except Exception as e:
                logger.error(f"Accuracy benchmark failed for {scenario_type}: {e}")

        result = bench.to_result()
        self.results[scenario_type] = result
        return result.metadata

    def benchmark_all_scenarios(
        self,
        evaluator,
        max_samples_per_scenario: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark accuracy across all scenarios.

        Args:
            evaluator: MRAGBenchEvaluator instance
            max_samples_per_scenario: Maximum samples per scenario

        Returns:
            Comprehensive accuracy benchmark results
        """
        scenarios = ["angle", "partial", "scope", "occlusion"]
        results = {}

        for scenario in scenarios:
            try:
                results[scenario] = self.benchmark_scenario_accuracy(
                    evaluator=evaluator,
                    scenario_type=scenario,
                    max_samples=max_samples_per_scenario
                )
            except Exception as e:
                logger.error(f"Failed to benchmark scenario {scenario}: {e}")
                results[scenario] = {"error": str(e)}

        # Calculate overall metrics
        total_questions = sum(r.get("total_questions", 0) for r in results.values() if "error" not in r)
        total_correct = sum(r.get("correct_answers", 0) for r in results.values() if "error" not in r)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

        return {
            "scenarios": results,
            "overall": {
                "total_questions": total_questions,
                "correct_answers": total_correct,
                "accuracy": overall_accuracy,
                "target_range": (0.53, 0.59),
                "target_achieved": 0.53 <= overall_accuracy <= 0.59
            }
        }


class LatencyBenchmarks:
    """
    Benchmarking suite for latency and throughput analysis.

    Measures timing characteristics under various conditions.
    """

    def __init__(self):
        """Initialize latency benchmarks."""
        self.results = {}

    def benchmark_latency_percentiles(
        self,
        operation: Callable,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark latency percentiles for an operation.

        Args:
            operation: Callable to benchmark
            num_iterations: Number of iterations to run

        Returns:
            Latency percentile statistics
        """
        latencies = []

        logger.info(f"Running {num_iterations} iterations for latency benchmark...")
        for i in range(num_iterations):
            start = time.time()
            try:
                operation()
                elapsed = time.time() - start
                latencies.append(elapsed)
            except Exception as e:
                logger.error(f"Operation failed in iteration {i}: {e}")

        if not latencies:
            return {"error": "All iterations failed"}

        return {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "min": min(latencies),
            "max": max(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "total_iterations": len(latencies)
        }

    def benchmark_throughput(
        self,
        operation: Callable,
        duration_seconds: float = 60.0
    ) -> Dict[str, float]:
        """
        Benchmark throughput over a time period.

        Args:
            operation: Callable to benchmark
            duration_seconds: How long to run the benchmark

        Returns:
            Throughput statistics
        """
        start_time = time.time()
        end_time = start_time + duration_seconds
        iterations = 0
        errors = 0

        logger.info(f"Running throughput benchmark for {duration_seconds}s...")
        while time.time() < end_time:
            try:
                operation()
                iterations += 1
            except Exception as e:
                errors += 1

        elapsed = time.time() - start_time
        throughput = iterations / elapsed

        return {
            "total_iterations": iterations,
            "total_errors": errors,
            "elapsed_time": elapsed,
            "throughput_qps": throughput,
            "avg_latency": elapsed / iterations if iterations > 0 else 0.0,
            "error_rate": errors / (iterations + errors) if (iterations + errors) > 0 else 0.0
        }


class BenchmarkOrchestrator:
    """
    Orchestrate all benchmarking activities for Sprint 8.

    Coordinates component, accuracy, and latency benchmarks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize benchmark orchestrator.

        Args:
            config: System configuration
        """
        self.config = config
        self.component_benchmarks = ComponentBenchmarks()
        self.accuracy_benchmarks = AccuracyBenchmarks()
        self.latency_benchmarks = LatencyBenchmarks()

    def run_comprehensive_benchmark(
        self,
        pipeline,
        evaluator,
        test_samples: List[Any],
        warmup: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.

        Args:
            pipeline: MRAG pipeline instance
            evaluator: MRAGBenchEvaluator instance
            test_samples: Test samples for benchmarking
            warmup: Whether to run warmup iterations

        Returns:
            Comprehensive benchmark report
        """
        logger.info("Starting comprehensive benchmark suite...")

        report = {
            "timestamp": time.time(),
            "config": self.config,
            "test_samples_count": len(test_samples),
            "component_benchmarks": {},
            "accuracy_benchmarks": {},
            "latency_benchmarks": {},
            "summary": {}
        }

        # Component benchmarks
        try:
            logger.info("Running component benchmarks...")

            # End-to-end benchmark
            e2e_suite = self.component_benchmarks.benchmark_end_to_end(
                pipeline=pipeline,
                test_samples=test_samples[:20],  # Limit to 20 samples
                warmup=2 if warmup else 0
            )

            report["component_benchmarks"] = self.component_benchmarks.generate_report()
        except Exception as e:
            logger.error(f"Component benchmarks failed: {e}")
            report["component_benchmarks"] = {"error": str(e)}

        # Accuracy benchmarks
        try:
            logger.info("Running accuracy benchmarks...")
            accuracy_results = self.accuracy_benchmarks.benchmark_all_scenarios(
                evaluator=evaluator,
                max_samples_per_scenario=50
            )
            report["accuracy_benchmarks"] = accuracy_results
        except Exception as e:
            logger.error(f"Accuracy benchmarks failed: {e}")
            report["accuracy_benchmarks"] = {"error": str(e)}

        # Generate summary
        report["summary"] = self._generate_summary(report)

        logger.info("Comprehensive benchmark complete")
        return report

    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            "timestamp": time.time(),
            "overall_performance": "unknown",
            "key_metrics": {},
            "bottlenecks": [],
            "recommendations": []
        }

        # Extract key metrics
        if "component_benchmarks" in report and "summary" in report["component_benchmarks"]:
            comp_summary = report["component_benchmarks"]["summary"]
            summary["key_metrics"]["avg_execution_time"] = comp_summary.get("avg_execution_time", 0.0)
            summary["key_metrics"]["p95_execution_time"] = comp_summary.get("p95_execution_time", 0.0)
            summary["key_metrics"]["peak_memory_usage"] = comp_summary.get("peak_memory_usage", 0.0)

            # Check if meeting targets
            if comp_summary.get("avg_execution_time", 100) < 30.0:
                summary["overall_performance"] = "good"
            elif comp_summary.get("avg_execution_time", 100) < 45.0:
                summary["overall_performance"] = "acceptable"
            else:
                summary["overall_performance"] = "needs_improvement"

        if "accuracy_benchmarks" in report and "overall" in report["accuracy_benchmarks"]:
            acc_summary = report["accuracy_benchmarks"]["overall"]
            summary["key_metrics"]["accuracy"] = acc_summary.get("accuracy", 0.0)
            summary["key_metrics"]["target_achieved"] = acc_summary.get("target_achieved", False)

        # Identify bottlenecks
        if summary["key_metrics"].get("avg_execution_time", 0) > 30.0:
            summary["bottlenecks"].append("Pipeline execution time exceeds 30s target")

        if summary["key_metrics"].get("peak_memory_usage", 0) > 15.0:
            summary["bottlenecks"].append("Memory usage exceeds 15GB limit")

        if not summary["key_metrics"].get("target_achieved", False):
            summary["bottlenecks"].append("Accuracy outside target range (53-59%)")

        # Generate recommendations
        if summary["bottlenecks"]:
            summary["recommendations"].append("Apply Sprint 8 optimizations to address identified bottlenecks")
            summary["recommendations"].append("Use PerformanceOptimizationOrchestrator for systematic improvements")
        else:
            summary["recommendations"].append("System performing within targets - focus on stability and edge cases")

        return summary
