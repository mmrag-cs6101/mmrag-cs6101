"""
Unit Tests for Sprint 8: Performance Optimization and Initial Accuracy Tuning

Tests all Sprint 8 components:
- Performance optimization module
- Benchmarking system
- Optimization orchestration
- Integration with existing system
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.evaluation.performance_optimizer import (
    PerformanceOptimizationOrchestrator,
    RetrievalOptimizer,
    GenerationOptimizer,
    MemoryOptimizer,
    PerformanceMetrics,
    OptimizationResult,
    OptimizationStrategy
)
from src.evaluation.benchmarking import (
    BenchmarkOrchestrator,
    ComponentBenchmarks,
    AccuracyBenchmarks,
    LatencyBenchmarks,
    PerformanceBenchmark,
    BenchmarkResult,
    BenchmarkSuite
)


# ========================================
# Performance Optimizer Tests
# ========================================

class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            avg_retrieval_time=5.0,
            avg_generation_time=25.0,
            avg_total_time=30.0,
            peak_memory_gb=15.0,
            accuracy=0.55
        )

        assert metrics.avg_retrieval_time == 5.0
        assert metrics.avg_generation_time == 25.0
        assert metrics.accuracy == 0.55

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = PerformanceMetrics()

        assert metrics.avg_retrieval_time == 0.0
        assert metrics.accuracy == 0.0
        assert metrics.throughput_qps == 0.0


class TestRetrievalOptimizer:
    """Test retrieval optimization functionality."""

    def test_initialization(self):
        """Test retrieval optimizer initialization."""
        config = {"top_k": 5, "batch_size": 16}
        optimizer = RetrievalOptimizer(config)

        assert optimizer.config == config
        assert optimizer.embedding_cache == {}
        assert optimizer.cache_hits == 0

    def test_embedding_caching(self):
        """Test embedding cache functionality."""
        optimizer = RetrievalOptimizer({})

        # First access - cache miss
        embeddings, stats = optimizer.optimize_embedding_generation(
            image_paths=["img1.jpg", "img2.jpg"],
            batch_size=2
        )

        assert stats["cache_misses"] == 2
        assert stats["cache_hits"] == 0
        assert stats["cache_hit_rate"] == 0.0

    def test_faiss_optimization(self):
        """Test FAISS search optimization."""
        optimizer = RetrievalOptimizer({})

        config = optimizer.optimize_faiss_search(top_k=5)

        assert "top_k" in config
        assert "nprobe" in config
        assert "use_gpu" in config
        assert config["top_k"] == 5

    def test_optimization_recommendations(self):
        """Test retrieval optimization recommendations."""
        optimizer = RetrievalOptimizer({})

        metrics = PerformanceMetrics(
            avg_retrieval_time=10.0,  # Above target
            embedding_cache_hit_rate=0.1  # Low
        )

        recommendations = optimizer.get_optimization_recommendations(metrics)

        assert len(recommendations) > 0
        assert any("batch size" in rec.lower() for rec in recommendations)


class TestGenerationOptimizer:
    """Test generation optimization functionality."""

    def test_initialization(self):
        """Test generation optimizer initialization."""
        config = {"temperature": 0.7, "max_length": 512}
        optimizer = GenerationOptimizer(config)

        assert optimizer.config == config
        assert "angle" in optimizer.prompt_templates
        assert "default" in optimizer.prompt_templates

    def test_prompt_optimization(self):
        """Test prompt engineering."""
        optimizer = GenerationOptimizer({})

        prompt = optimizer.optimize_prompt(
            question="What is shown in the image?",
            scenario_type="angle"
        )

        assert "angle" in prompt.lower() or "medical" in prompt.lower()
        assert "What is shown in the image?" in prompt

    def test_generation_params_low_accuracy(self):
        """Test parameter tuning for low accuracy."""
        optimizer = GenerationOptimizer({})

        params = optimizer.optimize_generation_params(
            current_accuracy=0.45,  # Below target
            target_accuracy_range=(0.53, 0.59)
        )

        # Should reduce randomness for better accuracy
        assert params["temperature"] < 0.5
        assert params["do_sample"] is False

    def test_generation_params_high_accuracy(self):
        """Test parameter tuning for high accuracy."""
        optimizer = GenerationOptimizer({})

        params = optimizer.optimize_generation_params(
            current_accuracy=0.65,  # Above target
            target_accuracy_range=(0.53, 0.59)
        )

        # Should add randomness to prevent overfitting
        assert params["temperature"] > 0.5

    def test_generation_params_in_range(self):
        """Test parameter tuning for accuracy in target range."""
        optimizer = GenerationOptimizer({})

        params = optimizer.optimize_generation_params(
            current_accuracy=0.56,  # In target range
            target_accuracy_range=(0.53, 0.59)
        )

        # Should fine-tune for stability
        assert 0.3 <= params["temperature"] <= 0.7

    def test_optimization_recommendations(self):
        """Test generation optimization recommendations."""
        optimizer = GenerationOptimizer({})

        metrics = PerformanceMetrics(
            avg_generation_time=30.0,  # Above target
            accuracy=0.45  # Below target
        )

        recommendations = optimizer.get_optimization_recommendations(metrics)

        assert len(recommendations) > 0
        assert any("temperature" in rec.lower() or "prompt" in rec.lower() for rec in recommendations)


class TestMemoryOptimizer:
    """Test memory optimization functionality."""

    def test_initialization(self):
        """Test memory optimizer initialization."""
        optimizer = MemoryOptimizer(memory_limit_gb=15.0)

        assert optimizer.memory_limit_gb == 15.0
        assert optimizer.memory_buffer_gb == 1.0
        assert optimizer.effective_limit_gb == 14.0

    def test_batch_size_critical_memory(self):
        """Test batch size optimization for critical memory."""
        optimizer = MemoryOptimizer(memory_limit_gb=15.0)

        new_batch_size = optimizer.optimize_batch_size(
            current_memory_gb=13.0,  # 93% of effective limit
            current_batch_size=8
        )

        assert new_batch_size < 8  # Should reduce

    def test_batch_size_low_memory(self):
        """Test batch size optimization for low memory."""
        optimizer = MemoryOptimizer(memory_limit_gb=15.0)

        new_batch_size = optimizer.optimize_batch_size(
            current_memory_gb=6.0,  # 43% of effective limit
            current_batch_size=8
        )

        assert new_batch_size > 8  # Should increase

    @pytest.mark.skipif(not pytest.importorskip("torch").cuda.is_available(),
                       reason="CUDA not available")
    def test_memory_clearing(self):
        """Test GPU memory clearing."""
        import torch

        optimizer = MemoryOptimizer()

        # Allocate some memory
        tensor = torch.randn(1000, 1000, device='cuda')

        # Clear memory
        stats = optimizer.clear_memory(aggressive=True)

        assert "before_gb" in stats
        assert "after_gb" in stats
        assert "freed_gb" in stats

    def test_optimization_recommendations(self):
        """Test memory optimization recommendations."""
        optimizer = MemoryOptimizer(memory_limit_gb=15.0)

        metrics = PerformanceMetrics(
            peak_memory_gb=16.0,  # Above limit
            memory_utilization_percent=90.0  # Critical
        )

        recommendations = optimizer.get_optimization_recommendations(metrics)

        assert len(recommendations) > 0
        assert any("memory" in rec.lower() for rec in recommendations)


class TestPerformanceOptimizationOrchestrator:
    """Test overall performance optimization orchestration."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        config = {
            "retrieval": {"top_k": 5},
            "generation": {"temperature": 0.7},
            "performance": {"memory_limit_gb": 15.0}
        }

        orchestrator = PerformanceOptimizationOrchestrator(config)

        assert orchestrator.config == config
        assert isinstance(orchestrator.retrieval_optimizer, RetrievalOptimizer)
        assert isinstance(orchestrator.generation_optimizer, GenerationOptimizer)
        assert isinstance(orchestrator.memory_optimizer, MemoryOptimizer)

    def test_default_target_metrics(self):
        """Test default target metrics."""
        orchestrator = PerformanceOptimizationOrchestrator({})

        targets = orchestrator.target_metrics

        assert targets.avg_retrieval_time == 5.0
        assert targets.avg_generation_time == 25.0
        assert targets.avg_total_time == 30.0
        assert targets.peak_memory_gb == 15.0
        assert 0.53 <= targets.accuracy <= 0.59

    def test_performance_analysis(self):
        """Test current performance analysis."""
        orchestrator = PerformanceOptimizationOrchestrator({})

        current_metrics = PerformanceMetrics(
            avg_retrieval_time=8.0,  # Above target
            avg_generation_time=30.0,  # Above target
            peak_memory_gb=16.0,  # Above target
            accuracy=0.45  # Below target
        )

        analysis = orchestrator.analyze_current_performance(current_metrics)

        assert "bottlenecks" in analysis
        assert len(analysis["bottlenecks"]) > 0
        assert "recommendations" in analysis
        assert "retrieval" in analysis["recommendations"]

    def test_apply_single_optimization(self):
        """Test applying single optimization strategy."""
        orchestrator = PerformanceOptimizationOrchestrator({})

        current_metrics = PerformanceMetrics(avg_retrieval_time=8.0)

        results = orchestrator.apply_optimizations(
            current_metrics=current_metrics,
            strategies=[OptimizationStrategy.RETRIEVAL]
        )

        assert len(results) == 1
        assert results[0].strategy == OptimizationStrategy.RETRIEVAL

    def test_apply_all_optimizations(self):
        """Test applying all optimization strategies."""
        orchestrator = PerformanceOptimizationOrchestrator({})

        current_metrics = PerformanceMetrics(
            avg_retrieval_time=8.0,
            avg_generation_time=30.0,
            peak_memory_gb=16.0,
            accuracy=0.45
        )

        results = orchestrator.apply_optimizations(
            current_metrics=current_metrics,
            strategies=None  # Apply all
        )

        assert len(results) == 5  # All strategies
        assert all(isinstance(r, OptimizationResult) for r in results)

    def test_optimization_report_generation(self):
        """Test comprehensive optimization report."""
        orchestrator = PerformanceOptimizationOrchestrator({})

        # Apply some optimizations
        current_metrics = PerformanceMetrics()
        orchestrator.apply_optimizations(current_metrics)

        report = orchestrator.generate_optimization_report()

        assert "total_optimizations" in report
        assert "successful_optimizations" in report
        assert "strategies_applied" in report
        assert "optimization_summary" in report


# ========================================
# Benchmarking Tests
# ========================================

class TestPerformanceBenchmark:
    """Test performance benchmark context manager."""

    def test_benchmark_timing(self):
        """Test benchmark timing functionality."""
        with PerformanceBenchmark("test_operation") as bench:
            time.sleep(0.1)  # Simulate work

        assert bench.elapsed_time >= 0.1
        assert bench.success is True
        assert bench.error is None

    def test_benchmark_failure(self):
        """Test benchmark with exception."""
        try:
            with PerformanceBenchmark("failing_operation") as bench:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert bench.success is False
        assert bench.error == "Test error"

    def test_benchmark_to_result(self):
        """Test converting benchmark to result."""
        with PerformanceBenchmark("test_op") as bench:
            bench.metadata["test_key"] = "test_value"

        result = bench.to_result()

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "test_op"
        assert result.metadata["test_key"] == "test_value"


class TestBenchmarkSuite:
    """Test benchmark suite functionality."""

    def test_suite_creation(self):
        """Test creating a benchmark suite."""
        suite = BenchmarkSuite(suite_name="test_suite")

        assert suite.suite_name == "test_suite"
        assert len(suite.results) == 0

    def test_adding_results(self):
        """Test adding results to suite."""
        suite = BenchmarkSuite(suite_name="test_suite")

        result = BenchmarkResult(
            benchmark_name="test1",
            execution_time=1.0,
            memory_usage_gb=5.0,
            success=True
        )

        suite.add_result(result)

        assert len(suite.results) == 1
        assert suite.results[0].benchmark_name == "test1"

    def test_suite_statistics(self):
        """Test suite statistical calculations."""
        suite = BenchmarkSuite(suite_name="test_suite")

        suite.add_result(BenchmarkResult("test1", 1.0, 5.0, True))
        suite.add_result(BenchmarkResult("test2", 2.0, 6.0, True))
        suite.add_result(BenchmarkResult("test3", 3.0, 7.0, False))

        assert suite.success_rate == 2/3
        assert suite.avg_execution_time == 2.0
        assert suite.avg_memory_usage == 6.0


class TestComponentBenchmarks:
    """Test component benchmarking suite."""

    def test_initialization(self):
        """Test component benchmarks initialization."""
        benchmarks = ComponentBenchmarks()

        assert benchmarks.results == {}

    def test_benchmark_report_generation(self):
        """Test generating benchmark report."""
        benchmarks = ComponentBenchmarks()

        # Add some mock results
        suite = BenchmarkSuite(suite_name="test_component")
        suite.add_result(BenchmarkResult("test1", 1.0, 5.0, True))
        suite.finalize()

        benchmarks.results["test_component"] = suite

        report = benchmarks.generate_report()

        assert "components" in report
        assert "summary" in report
        assert "test_component" in report["components"]


class TestAccuracyBenchmarks:
    """Test accuracy benchmarking functionality."""

    def test_initialization(self):
        """Test accuracy benchmarks initialization."""
        benchmarks = AccuracyBenchmarks()

        assert benchmarks.results == {}


class TestLatencyBenchmarks:
    """Test latency benchmarking functionality."""

    def test_initialization(self):
        """Test latency benchmarks initialization."""
        benchmarks = LatencyBenchmarks()

        assert benchmarks.results == {}

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        benchmarks = LatencyBenchmarks()

        # Mock operation that takes ~0.01s
        def mock_operation():
            time.sleep(0.01)

        results = benchmarks.benchmark_latency_percentiles(
            operation=mock_operation,
            num_iterations=10
        )

        assert "mean" in results
        assert "median" in results
        assert "p95" in results
        assert "p99" in results
        assert results["total_iterations"] == 10

    def test_throughput_measurement(self):
        """Test throughput measurement."""
        benchmarks = LatencyBenchmarks()

        # Mock fast operation
        def mock_operation():
            time.sleep(0.001)

        results = benchmarks.benchmark_throughput(
            operation=mock_operation,
            duration_seconds=0.1
        )

        assert "throughput_qps" in results
        assert "avg_latency" in results
        assert "error_rate" in results
        assert results["total_iterations"] > 0


class TestBenchmarkOrchestrator:
    """Test comprehensive benchmark orchestration."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        config = {"test": "config"}
        orchestrator = BenchmarkOrchestrator(config)

        assert orchestrator.config == config
        assert isinstance(orchestrator.component_benchmarks, ComponentBenchmarks)
        assert isinstance(orchestrator.accuracy_benchmarks, AccuracyBenchmarks)
        assert isinstance(orchestrator.latency_benchmarks, LatencyBenchmarks)


# ========================================
# Integration Tests
# ========================================

class TestSprint8Integration:
    """Test Sprint 8 component integration."""

    def test_optimization_and_benchmarking_integration(self):
        """Test integration between optimization and benchmarking."""
        # Create optimizer
        config = {"performance": {"memory_limit_gb": 15.0}}
        optimizer = PerformanceOptimizationOrchestrator(config)

        # Apply optimizations
        metrics = PerformanceMetrics(
            avg_retrieval_time=8.0,
            accuracy=0.45
        )

        results = optimizer.apply_optimizations(metrics)

        # Verify results can be used for benchmarking
        assert len(results) > 0
        assert all(hasattr(r, "improvements") for r in results)

    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow."""
        # 1. Create baseline metrics
        baseline = PerformanceMetrics(
            avg_retrieval_time=8.0,
            avg_generation_time=30.0,
            peak_memory_gb=16.0,
            accuracy=0.45
        )

        # 2. Create orchestrator
        orchestrator = PerformanceOptimizationOrchestrator({})

        # 3. Analyze performance
        analysis = orchestrator.analyze_current_performance(baseline)

        assert len(analysis["bottlenecks"]) > 0

        # 4. Apply optimizations
        results = orchestrator.apply_optimizations(baseline)

        assert len(results) > 0

        # 5. Generate report
        report = orchestrator.generate_optimization_report()

        assert "total_optimizations" in report


# ========================================
# Test Fixtures
# ========================================

@pytest.fixture
def sample_metrics():
    """Provide sample performance metrics."""
    return PerformanceMetrics(
        avg_retrieval_time=5.0,
        avg_generation_time=25.0,
        avg_total_time=30.0,
        peak_memory_gb=15.0,
        accuracy=0.55,
        error_rate=0.03
    )


@pytest.fixture
def sample_config():
    """Provide sample configuration."""
    return {
        "retrieval": {
            "top_k": 5,
            "batch_size": 16
        },
        "generation": {
            "temperature": 0.7,
            "max_length": 512
        },
        "performance": {
            "memory_limit_gb": 15.0
        }
    }


# Run tests with: pytest tests/test_sprint8_optimization.py -v
