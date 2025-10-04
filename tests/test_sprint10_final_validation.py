"""
Tests for Sprint 10: Final Accuracy Validation

Comprehensive test suite for final validation framework including:
- FinalAccuracyValidator functionality
- Multi-run statistical validation
- System performance metrics
- Production readiness assessment
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

# Import Sprint 10 components
from src.evaluation.final_validator import (
    FinalAccuracyValidator,
    FinalValidationResults,
    ScenarioFinalResults,
    SystemPerformanceMetrics,
    MultiRunStatistics
)
from src.evaluation.evaluator import ScenarioMetrics, PerspectiveChangeType
from src.config import MRAGConfig


# Fixtures

@pytest.fixture
def mock_config():
    """Create mock MRAG configuration."""
    config = Mock(spec=MRAGConfig)
    config.dataset = Mock()
    config.dataset.data_path = "/tmp/test_data"
    config.dataset.batch_size = 4
    config.dataset.image_size = (224, 224)

    config.model = Mock()
    config.model.vlm_name = "test-vlm"
    config.model.retriever_name = "test-retriever"
    config.model.quantization = "4bit"

    config.retrieval = Mock()
    config.retrieval.top_k = 5
    config.retrieval.embedding_dim = 512

    config.generation = Mock()
    config.generation.max_length = 256
    config.generation.temperature = 0.3
    config.generation.top_p = 0.9

    config.performance = Mock()
    config.performance.memory_limit_gb = 15.0
    config.performance.memory_buffer_gb = 1.0

    return config


@pytest.fixture
def mock_scenario_metrics():
    """Create mock scenario metrics."""
    def create_metrics(
        scenario: str,
        accuracy: float,
        total: int = 100
    ) -> ScenarioMetrics:
        correct = int(accuracy * total)
        return ScenarioMetrics(
            scenario_type=scenario,
            total_questions=total,
            correct_answers=correct,
            accuracy=accuracy,
            avg_processing_time=20.0,
            avg_retrieval_time=3.0,
            avg_generation_time=17.0,
            confidence_scores=[0.85] * total,
            error_count=0,
            error_rate=0.0
        )
    return create_metrics


@pytest.fixture
def validator(mock_config, tmp_path):
    """Create validator instance for testing."""
    with patch('src.evaluation.final_validator.MRAGBenchEvaluator'):
        with patch('src.evaluation.final_validator.MemoryManager'):
            validator = FinalAccuracyValidator(
                config=mock_config,
                target_range=(0.53, 0.59),
                output_dir=str(tmp_path / "sprint10")
            )
            yield validator


# Test Data Structures

class TestDataStructures:
    """Test Sprint 10 data structures."""

    def test_multi_run_statistics_creation(self):
        """Test MultiRunStatistics dataclass creation."""
        stats = MultiRunStatistics(
            mean_accuracy=0.55,
            std_accuracy=0.02,
            median_accuracy=0.56,
            min_accuracy=0.52,
            max_accuracy=0.58,
            confidence_interval_95=(0.53, 0.57),
            coefficient_of_variation=0.04,
            individual_run_accuracies=[0.54, 0.55, 0.56],
            statistical_significance=True,
            p_value=0.03
        )

        assert stats.mean_accuracy == 0.55
        assert stats.std_accuracy == 0.02
        assert len(stats.individual_run_accuracies) == 3
        assert stats.statistical_significance is True

    def test_scenario_final_results_creation(self):
        """Test ScenarioFinalResults dataclass creation."""
        multi_run_stats = MultiRunStatistics(
            mean_accuracy=0.56,
            std_accuracy=0.01,
            median_accuracy=0.56,
            min_accuracy=0.55,
            max_accuracy=0.57,
            confidence_interval_95=(0.55, 0.57),
            coefficient_of_variation=0.02,
            individual_run_accuracies=[0.55, 0.56, 0.57],
            statistical_significance=True
        )

        result = ScenarioFinalResults(
            scenario_type="angle",
            total_samples=322,
            correct_answers=180,
            accuracy=0.56,
            confidence_interval_95=(0.53, 0.59),
            multi_run_stats=multi_run_stats,
            avg_processing_time=20.0,
            avg_retrieval_time=3.0,
            avg_generation_time=17.0,
            avg_confidence_score=0.85,
            std_confidence_score=0.10,
            error_rate=0.01,
            in_target_range=True,
            target_range=(0.53, 0.59)
        )

        assert result.scenario_type == "angle"
        assert result.accuracy == 0.56
        assert result.in_target_range is True
        assert result.multi_run_stats is not None

    def test_system_performance_metrics_creation(self):
        """Test SystemPerformanceMetrics dataclass creation."""
        metrics = SystemPerformanceMetrics(
            avg_query_time=22.0,
            p50_query_time=20.0,
            p95_query_time=28.0,
            p99_query_time=32.0,
            total_evaluation_time=1800.0,
            peak_memory_gb=14.5,
            avg_memory_gb=13.0,
            memory_utilization_percent=96.7,
            memory_within_limit=True,
            total_queries=778,
            successful_queries=775,
            failed_queries=3,
            success_rate=0.996,
            error_rate=0.004,
            queries_per_second=0.43,
            samples_per_minute=25.9
        )

        assert metrics.avg_query_time == 22.0
        assert metrics.peak_memory_gb == 14.5
        assert metrics.success_rate == 0.996
        assert metrics.memory_within_limit is True


# Test Validator Initialization

class TestValidatorInitialization:
    """Test validator initialization."""

    def test_validator_initialization(self, validator):
        """Test validator initializes correctly."""
        assert validator.target_range == (0.53, 0.59)
        assert validator.output_dir.exists()
        assert validator.evaluation_runs == []
        assert validator.performance_history == []

    def test_validator_with_custom_target(self, mock_config, tmp_path):
        """Test validator with custom target range."""
        with patch('src.evaluation.final_validator.MRAGBenchEvaluator'):
            with patch('src.evaluation.final_validator.MemoryManager'):
                validator = FinalAccuracyValidator(
                    config=mock_config,
                    target_range=(0.50, 0.65),
                    output_dir=str(tmp_path / "custom")
                )

                assert validator.target_range == (0.50, 0.65)


# Test Statistical Methods

class TestStatisticalMethods:
    """Test statistical calculation methods."""

    def test_confidence_interval_calculation(self, validator):
        """Test confidence interval calculation."""
        # Wilson score interval test
        ci = validator._calculate_confidence_interval(
            accuracy=0.55,
            sample_count=100
        )

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert 0.0 <= ci[0] <= ci[1] <= 1.0
        assert ci[0] < 0.55 < ci[1]  # Accuracy should be within CI

    def test_confidence_interval_edge_cases(self, validator):
        """Test confidence interval edge cases."""
        # Zero samples
        ci = validator._calculate_confidence_interval(0.5, 0)
        assert ci == (0.0, 0.0)

        # Perfect accuracy
        ci = validator._calculate_confidence_interval(1.0, 100)
        assert ci[1] == 1.0

        # Zero accuracy
        ci = validator._calculate_confidence_interval(0.0, 100)
        assert ci[0] == 0.0

    def test_multi_run_statistics_calculation(self, validator):
        """Test multi-run statistics calculation."""
        accuracies = [0.54, 0.55, 0.56, 0.54, 0.57]
        target_range = (0.53, 0.59)

        stats = validator._calculate_multi_run_statistics(
            accuracies,
            target_range
        )

        assert stats.mean_accuracy == pytest.approx(0.552, abs=0.01)
        assert stats.std_accuracy > 0
        assert stats.min_accuracy == 0.54
        assert stats.max_accuracy == 0.57
        assert len(stats.individual_run_accuracies) == 5

    def test_multi_run_statistics_empty_list(self, validator):
        """Test multi-run statistics with empty list."""
        stats = validator._calculate_multi_run_statistics([], (0.53, 0.59))

        assert stats.mean_accuracy == 0.0
        assert stats.std_accuracy == 0.0
        assert len(stats.individual_run_accuracies) == 0

    def test_consistency_calculation(self, validator):
        """Test consistency metric calculation."""
        # Perfect consistency
        consistency = validator._calculate_consistency([0.55] * 5)
        assert consistency == pytest.approx(1.0, abs=0.01)

        # Moderate variance
        consistency = validator._calculate_consistency([0.50, 0.55, 0.60])
        assert 0.0 < consistency < 1.0

        # Empty list
        consistency = validator._calculate_consistency([])
        assert consistency == 1.0

        # Single value
        consistency = validator._calculate_consistency([0.55])
        assert consistency == 1.0


# Test Evaluation Methods

class TestEvaluationMethods:
    """Test evaluation execution methods."""

    def test_run_single_evaluation(self, validator, mock_scenario_metrics):
        """Test single evaluation run."""
        # Mock evaluator.evaluate_scenario
        validator.evaluator.evaluate_scenario = Mock(side_effect=[
            mock_scenario_metrics("angle", 0.56, 322),
            mock_scenario_metrics("partial", 0.54, 246),
            mock_scenario_metrics("scope", 0.52, 102),
            mock_scenario_metrics("occlusion", 0.50, 108)
        ])

        # Mock memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.gpu_allocated_gb = 14.5
        validator.memory_manager.monitor.get_current_stats = Mock(
            return_value=mock_memory_stats
        )

        # Run evaluation
        result = validator._run_single_evaluation(max_samples_per_scenario=None)

        # Verify results
        assert "scenario_results" in result
        assert "overall_accuracy" in result
        assert "total_questions" in result
        assert result["total_questions"] == 778  # 322+246+102+108
        assert 0.0 <= result["overall_accuracy"] <= 1.0

    def test_run_multiple_evaluations(self, validator, mock_scenario_metrics):
        """Test multiple evaluation runs."""
        # Mock single evaluation
        validator._run_single_evaluation = Mock(side_effect=[
            {
                "overall_accuracy": 0.54,
                "total_questions": 100,
                "total_correct": 54,
                "scenario_results": {},
                "evaluation_time": 60.0,
                "avg_processing_time": 20.0,
                "avg_retrieval_time": 3.0,
                "avg_generation_time": 17.0,
                "avg_confidence_score": 0.85,
                "peak_memory_gb": 14.5,
                "processing_times": [20.0] * 100,
                "confidence_scores": [0.85] * 100
            }
            for _ in range(3)
        ])

        # Mock memory manager
        validator.memory_manager.clear_gpu_memory = Mock()

        # Run multiple evaluations
        results = validator._run_multiple_evaluations(
            num_runs=3,
            max_samples_per_scenario=None
        )

        assert len(results) == 3
        assert all("run_number" in r for r in results)
        assert validator.memory_manager.clear_gpu_memory.call_count == 3


# Test Result Aggregation

class TestResultAggregation:
    """Test result aggregation methods."""

    def test_aggregate_multi_run_results(self, validator, mock_scenario_metrics):
        """Test aggregation of multiple evaluation runs."""
        # Create mock run results
        run_results = [
            {
                "overall_accuracy": 0.54,
                "total_questions": 778,
                "total_correct": 420,
                "scenario_results": {
                    "angle": mock_scenario_metrics("angle", 0.56, 322),
                    "partial": mock_scenario_metrics("partial", 0.54, 246),
                    "scope": mock_scenario_metrics("scope", 0.52, 102),
                    "occlusion": mock_scenario_metrics("occlusion", 0.50, 108)
                },
                "evaluation_time": 600.0,
                "avg_processing_time": 20.0,
                "avg_retrieval_time": 3.0,
                "avg_generation_time": 17.0,
                "avg_confidence_score": 0.85,
                "peak_memory_gb": 14.5,
                "processing_times": [20.0] * 778,
                "confidence_scores": [0.85] * 778
            }
            for _ in range(3)
        ]

        # Aggregate results
        final_results = validator._aggregate_multi_run_results(run_results)

        # Verify overall metrics
        assert final_results.overall_accuracy == pytest.approx(0.54, abs=0.01)
        assert final_results.total_questions == 778
        assert final_results.num_evaluation_runs == 3

        # Verify scenario results
        assert len(final_results.scenario_results) == 4
        assert "angle" in final_results.scenario_results
        assert "partial" in final_results.scenario_results

        # Verify multi-run statistics
        assert final_results.multi_run_statistics is not None
        assert final_results.cross_run_consistency > 0.0

    def test_create_scenario_final_result(self, validator, mock_scenario_metrics):
        """Test creation of scenario final result."""
        # Create metrics from multiple runs
        metrics_list = [
            mock_scenario_metrics("angle", 0.55, 322),
            mock_scenario_metrics("angle", 0.56, 322),
            mock_scenario_metrics("angle", 0.54, 322)
        ]

        # Create final result
        final_result = validator._create_scenario_final_result(
            "angle",
            metrics_list,
            num_runs=3
        )

        # Verify
        assert final_result.scenario_type == "angle"
        assert final_result.total_samples == 322
        assert 0.54 <= final_result.accuracy <= 0.56
        assert final_result.multi_run_stats is not None
        assert len(final_result.confidence_interval_95) == 2


# Test Performance Metrics Collection

class TestPerformanceMetrics:
    """Test performance metrics collection."""

    def test_collect_performance_metrics(self, validator):
        """Test performance metrics collection."""
        run_results = [
            {
                "processing_times": [20.0, 22.0, 18.0, 25.0],
                "evaluation_time": 100.0,
                "peak_memory_gb": 14.5,
                "total_questions": 100,
                "total_correct": 55,
                "confidence_scores": [0.85] * 100
            },
            {
                "processing_times": [21.0, 23.0, 19.0, 26.0],
                "evaluation_time": 105.0,
                "peak_memory_gb": 14.8,
                "total_questions": 100,
                "total_correct": 56,
                "confidence_scores": [0.86] * 100
            }
        ]

        metrics = validator._collect_performance_metrics(run_results)

        assert isinstance(metrics, SystemPerformanceMetrics)
        assert metrics.avg_query_time > 0
        assert metrics.peak_memory_gb > 0
        assert metrics.total_queries == 200
        assert metrics.success_rate > 0
        assert metrics.queries_per_second > 0

    def test_performance_metrics_timing_percentiles(self, validator):
        """Test timing percentile calculations."""
        run_results = [{
            "processing_times": list(range(1, 101)),  # 1-100 seconds
            "evaluation_time": 1000.0,
            "peak_memory_gb": 14.0,
            "total_questions": 100,
            "total_correct": 50,
            "confidence_scores": [0.8] * 100
        }]

        metrics = validator._collect_performance_metrics(run_results)

        assert metrics.p50_query_time == pytest.approx(50.5, abs=1.0)
        assert metrics.p95_query_time == pytest.approx(95.0, abs=1.0)
        assert metrics.p99_query_time == pytest.approx(99.0, abs=1.0)


# Test Assessment Methods

class TestAssessmentMethods:
    """Test assessment and recommendation methods."""

    def test_assess_statistical_confidence_high(self, validator):
        """Test statistical confidence assessment - high confidence."""
        results = FinalValidationResults(
            overall_accuracy=0.55,
            overall_confidence_interval=(0.53, 0.57),
            total_questions=778,
            total_correct=428,
            target_achieved=True,
            target_range=(0.53, 0.59),
            scenario_results={},
            scenarios_in_target=4,
            scenario_consistency=0.95,
            num_evaluation_runs=3,
            multi_run_statistics=None,
            cross_run_consistency=0.97,
            performance_metrics=None,
            statistical_confidence="",
            baseline_comparison=None,
            recommendations=[],
            production_readiness="",
            timestamp="",
            total_validation_time=0.0,
            configuration_summary={}
        )

        confidence = validator._assess_statistical_confidence(results)
        assert confidence in ["high", "medium", "low"]

    def test_assess_production_readiness_ready(self, validator):
        """Test production readiness - system ready."""
        perf_metrics = SystemPerformanceMetrics(
            avg_query_time=25.0,  # Within 30s target
            p50_query_time=22.0,
            p95_query_time=28.0,
            p99_query_time=30.0,
            total_evaluation_time=1800.0,
            peak_memory_gb=14.0,  # Within 15GB limit
            avg_memory_gb=13.0,
            memory_utilization_percent=93.3,
            memory_within_limit=True,
            total_queries=778,
            successful_queries=775,
            failed_queries=3,
            success_rate=0.996,  # > 95%
            error_rate=0.004,
            queries_per_second=0.43,
            samples_per_minute=25.9
        )

        results = FinalValidationResults(
            overall_accuracy=0.55,  # In target
            overall_confidence_interval=(0.53, 0.57),
            total_questions=778,
            total_correct=428,
            target_achieved=True,
            target_range=(0.53, 0.59),
            scenario_results={},
            scenarios_in_target=4,
            scenario_consistency=0.92,
            num_evaluation_runs=3,
            multi_run_statistics=None,
            cross_run_consistency=0.95,
            performance_metrics=perf_metrics,
            statistical_confidence="high",
            baseline_comparison=None,
            recommendations=[],
            production_readiness="",
            timestamp="",
            total_validation_time=0.0,
            configuration_summary={}
        )

        readiness = validator._assess_production_readiness(results)
        assert readiness == "ready"

    def test_assess_production_readiness_needs_optimization(self, validator):
        """Test production readiness - needs optimization."""
        perf_metrics = SystemPerformanceMetrics(
            avg_query_time=32.0,  # Slightly over 30s target
            p50_query_time=30.0,
            p95_query_time=35.0,
            p99_query_time=40.0,
            total_evaluation_time=2000.0,
            peak_memory_gb=14.5,
            avg_memory_gb=13.5,
            memory_utilization_percent=96.7,
            memory_within_limit=True,
            total_queries=778,
            successful_queries=775,
            failed_queries=3,
            success_rate=0.996,
            error_rate=0.004,
            queries_per_second=0.39,
            samples_per_minute=23.3
        )

        results = FinalValidationResults(
            overall_accuracy=0.55,
            overall_confidence_interval=(0.53, 0.57),
            total_questions=778,
            total_correct=428,
            target_achieved=True,
            target_range=(0.53, 0.59),
            scenario_results={},
            scenarios_in_target=3,  # Not all scenarios in target
            scenario_consistency=0.85,
            num_evaluation_runs=3,
            multi_run_statistics=None,
            cross_run_consistency=0.92,
            performance_metrics=perf_metrics,
            statistical_confidence="medium",
            baseline_comparison=None,
            recommendations=[],
            production_readiness="",
            timestamp="",
            total_validation_time=0.0,
            configuration_summary={}
        )

        readiness = validator._assess_production_readiness(results)
        assert readiness in ["needs_optimization", "ready"]

    def test_generate_final_recommendations(self, validator):
        """Test final recommendation generation."""
        perf_metrics = SystemPerformanceMetrics(
            avg_query_time=22.0,
            p50_query_time=20.0,
            p95_query_time=28.0,
            p99_query_time=32.0,
            total_evaluation_time=1800.0,
            peak_memory_gb=14.5,
            avg_memory_gb=13.0,
            memory_utilization_percent=96.7,
            memory_within_limit=True,
            total_queries=778,
            successful_queries=775,
            failed_queries=3,
            success_rate=0.996,
            error_rate=0.004,
            queries_per_second=0.43,
            samples_per_minute=25.9
        )

        results = FinalValidationResults(
            overall_accuracy=0.55,
            overall_confidence_interval=(0.53, 0.57),
            total_questions=778,
            total_correct=428,
            target_achieved=True,
            target_range=(0.53, 0.59),
            scenario_results={},
            scenarios_in_target=4,
            scenario_consistency=0.92,
            num_evaluation_runs=3,
            multi_run_statistics=None,
            cross_run_consistency=0.95,
            performance_metrics=perf_metrics,
            statistical_confidence="high",
            baseline_comparison=None,
            recommendations=[],
            production_readiness="ready",
            timestamp="",
            total_validation_time=0.0,
            configuration_summary={}
        )

        recommendations = validator._generate_final_recommendations(results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("✓" in rec or "accuracy" in rec.lower() for rec in recommendations)


# Test Configuration Methods

class TestConfigurationMethods:
    """Test configuration summary methods."""

    def test_get_config_summary(self, validator):
        """Test configuration summary generation."""
        summary = validator._get_config_summary()

        assert "model" in summary
        assert "retrieval" in summary
        assert "generation" in summary
        assert "performance" in summary

        assert summary["model"]["vlm"] == "test-vlm"
        assert summary["retrieval"]["top_k"] == 5
        assert summary["generation"]["temperature"] == 0.3


# Integration Tests

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_comprehensive_validation_workflow(self, validator, mock_scenario_metrics, tmp_path):
        """Test complete validation workflow."""
        # Mock evaluator initialization and evaluation
        validator.evaluator.initialize_components = Mock()
        validator.evaluator.evaluate_scenario = Mock(side_effect=[
            # Run 1
            mock_scenario_metrics("angle", 0.56, 322),
            mock_scenario_metrics("partial", 0.54, 246),
            mock_scenario_metrics("scope", 0.52, 102),
            mock_scenario_metrics("occlusion", 0.50, 108),
            # Run 2
            mock_scenario_metrics("angle", 0.55, 322),
            mock_scenario_metrics("partial", 0.53, 246),
            mock_scenario_metrics("scope", 0.51, 102),
            mock_scenario_metrics("occlusion", 0.49, 108),
        ])

        # Mock memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.gpu_allocated_gb = 14.5
        validator.memory_manager.monitor.get_current_stats = Mock(
            return_value=mock_memory_stats
        )

        # Mock memory clear
        validator.memory_manager.clear_gpu_memory = Mock()

        # Mock cleanup
        validator.evaluator.cleanup = Mock()
        validator.memory_manager.emergency_cleanup = Mock()

        # Run validation with 2 runs (reduced for testing)
        results = validator.run_comprehensive_validation(
            num_runs=2,
            full_dataset=False,
            max_samples_per_scenario=None
        )

        # Verify results structure
        assert isinstance(results, FinalValidationResults)
        assert results.num_evaluation_runs == 2
        assert results.total_questions == 778
        assert len(results.scenario_results) == 4
        assert results.performance_metrics is not None
        assert results.statistical_confidence in ["high", "medium", "low"]
        assert results.production_readiness in ["ready", "needs_optimization", "not_ready"]
        assert len(results.recommendations) > 0

        # Verify results saved
        json_file = tmp_path / "sprint10" / "sprint10_final_validation_results.json"
        summary_file = tmp_path / "sprint10" / "sprint10_summary_report.md"
        assert json_file.exists()
        assert summary_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
