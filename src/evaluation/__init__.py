"""
Evaluation Module

Provides MRAG-Bench evaluation framework for accuracy calculation and performance analysis.
Implements evaluation methodology matching the original paper.

Sprint 8 adds performance optimization and comprehensive benchmarking capabilities.
Sprint 10 adds final accuracy validation with multi-run statistical analysis.
"""

from .results import EvaluationResults
from .evaluator import MRAGBenchEvaluator, PerspectiveChangeType, ScenarioMetrics, EvaluationSession

# Sprint 8: Performance Optimization and Benchmarking
from .performance_optimizer import (
    PerformanceOptimizationOrchestrator,
    RetrievalOptimizer,
    GenerationOptimizer,
    MemoryOptimizer,
    PerformanceMetrics,
    OptimizationResult,
    OptimizationStrategy
)
from .benchmarking import (
    BenchmarkOrchestrator,
    ComponentBenchmarks,
    AccuracyBenchmarks,
    LatencyBenchmarks,
    PerformanceBenchmark,
    BenchmarkResult,
    BenchmarkSuite
)

# Sprint 10: Final Accuracy Validation
from .final_validator import (
    FinalAccuracyValidator,
    FinalValidationResults,
    ScenarioFinalResults,
    SystemPerformanceMetrics,
    MultiRunStatistics
)

__all__ = [
    # Core evaluation
    "EvaluationResults",
    "MRAGBenchEvaluator",
    "PerspectiveChangeType",
    "ScenarioMetrics",
    "EvaluationSession",
    # Sprint 8: Performance optimization
    "PerformanceOptimizationOrchestrator",
    "RetrievalOptimizer",
    "GenerationOptimizer",
    "MemoryOptimizer",
    "PerformanceMetrics",
    "OptimizationResult",
    "OptimizationStrategy",
    # Sprint 8: Benchmarking
    "BenchmarkOrchestrator",
    "ComponentBenchmarks",
    "AccuracyBenchmarks",
    "LatencyBenchmarks",
    "PerformanceBenchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
    # Sprint 10: Final validation
    "FinalAccuracyValidator",
    "FinalValidationResults",
    "ScenarioFinalResults",
    "SystemPerformanceMetrics",
    "MultiRunStatistics",
]