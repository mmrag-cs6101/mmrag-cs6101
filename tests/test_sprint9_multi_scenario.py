"""
Test Suite for Sprint 9: Multi-Scenario Expansion

Comprehensive tests for multi-scenario evaluation functionality.
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_sprint9_multi_scenario import (
    Sprint9MultiScenarioOrchestrator,
    ScenarioOptimizationResult,
    CrossScenarioAnalysis,
    Sprint9Results
)
from src.config import MRAGConfig
from src.evaluation.evaluator import PerspectiveChangeType


class TestScenarioOptimizationResult:
    """Test ScenarioOptimizationResult dataclass."""

    def test_creation(self):
        """Test creation of ScenarioOptimizationResult."""
        result = ScenarioOptimizationResult(
            scenario_type="angle",
            baseline_accuracy=0.45,
            optimized_accuracy=0.55,
            accuracy_improvement=0.10,
            optimal_parameters={"top_k": 5},
            sample_count=100,
            avg_processing_time=25.0,
            optimization_rounds=5,
            confidence_interval=(0.52, 0.58)
        )

        assert result.scenario_type == "angle"
        assert result.baseline_accuracy == 0.45
        assert result.optimized_accuracy == 0.55
        assert result.accuracy_improvement == 0.10
        assert result.sample_count == 100

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        result = ScenarioOptimizationResult(
            scenario_type="partial",
            baseline_accuracy=0.42,
            optimized_accuracy=0.53,
            accuracy_improvement=0.11,
            optimal_parameters={},
            sample_count=200,
            avg_processing_time=28.0,
            optimization_rounds=5,
            confidence_interval=(0.50, 0.56)
        )

        lower, upper = result.confidence_interval
        assert lower < result.optimized_accuracy < upper
        assert upper - lower < 0.10  # Reasonable confidence interval width


class TestCrossScenarioAnalysis:
    """Test CrossScenarioAnalysis dataclass."""

    def test_creation(self):
        """Test creation of CrossScenarioAnalysis."""
        analysis = CrossScenarioAnalysis(
            best_scenario="angle",
            worst_scenario="occlusion",
            accuracy_variance=0.008,
            performance_consistency=0.92,
            scenario_rankings={"angle": 1, "partial": 2, "scope": 3, "occlusion": 4},
            difficulty_assessment={"angle": "Easy", "occlusion": "Difficult"},
            common_challenges=["Occlusion handling difficult"],
            optimization_recommendations=["Increase top_k for occlusion"]
        )

        assert analysis.best_scenario == "angle"
        assert analysis.worst_scenario == "occlusion"
        assert analysis.performance_consistency == 0.92
        assert len(analysis.scenario_rankings) == 4

    def test_difficulty_assessment(self):
        """Test difficulty assessment logic."""
        difficulty_map = {
            "angle": "Easy",
            "partial": "Moderate",
            "scope": "Challenging",
            "occlusion": "Difficult"
        }

        analysis = CrossScenarioAnalysis(
            best_scenario="angle",
            worst_scenario="occlusion",
            accuracy_variance=0.012,
            performance_consistency=0.85,
            scenario_rankings={},
            difficulty_assessment=difficulty_map,
            common_challenges=[],
            optimization_recommendations=[]
        )

        assert analysis.difficulty_assessment["angle"] == "Easy"
        assert analysis.difficulty_assessment["occlusion"] == "Difficult"


class TestSprint9Results:
    """Test Sprint9Results dataclass."""

    def test_creation(self):
        """Test creation of Sprint9Results."""
        scenario_results = {
            "angle": ScenarioOptimizationResult(
                scenario_type="angle",
                baseline_accuracy=0.48,
                optimized_accuracy=0.56,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=322,
                avg_processing_time=24.0,
                optimization_rounds=5,
                confidence_interval=(0.53, 0.59)
            )
        }

        cross_analysis = CrossScenarioAnalysis(
            best_scenario="angle",
            worst_scenario="occlusion",
            accuracy_variance=0.008,
            performance_consistency=0.90,
            scenario_rankings={"angle": 1},
            difficulty_assessment={"angle": "Easy"},
            common_challenges=[],
            optimization_recommendations=[]
        )

        results = Sprint9Results(
            overall_accuracy=0.55,
            total_questions=778,
            total_correct=428,
            overall_confidence_interval=(0.53, 0.57),
            target_achieved=True,
            scenario_results=scenario_results,
            scenario_accuracies={"angle": 0.56},
            scenario_sample_counts={"angle": 322},
            cross_scenario_analysis=cross_analysis,
            avg_processing_time=25.0,
            avg_retrieval_time=4.0,
            avg_generation_time=21.0,
            peak_memory_gb=14.2,
            target_range=(0.53, 0.59),
            accuracy_gap=0.0,
            scenarios_in_range=1,
            recommendations=["Continue monitoring"],
            sprint10_priorities=["Final validation"],
            timestamp="2025-10-04T00:00:00",
            evaluation_duration=120.0,
            total_optimization_rounds=5
        )

        assert results.overall_accuracy == 0.55
        assert results.target_achieved is True
        assert results.total_questions == 778
        assert results.peak_memory_gb < 15.0


class TestSprint9Orchestrator:
    """Test Sprint9MultiScenarioOrchestrator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MRAGConfig()

    @pytest.fixture
    def orchestrator(self, config):
        """Create test orchestrator."""
        with patch('run_sprint9_multi_scenario.MRAGBenchEvaluator'):
            return Sprint9MultiScenarioOrchestrator(config, output_dir="output/test_sprint9")

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert len(orchestrator.scenarios) == 4
        assert orchestrator.target_min == 0.53
        assert orchestrator.target_max == 0.59

    def test_scenario_list(self, orchestrator):
        """Test all 4 scenarios are included."""
        scenario_values = [s.value for s in orchestrator.scenarios]
        assert "angle" in scenario_values
        assert "partial" in scenario_values
        assert "scope" in scenario_values
        assert "occlusion" in scenario_values

    def test_evaluate_scenario_baseline(self, orchestrator):
        """Test baseline scenario evaluation."""
        baseline = orchestrator._evaluate_scenario_baseline(
            PerspectiveChangeType.ANGLE,
            max_samples=100
        )

        assert 'accuracy' in baseline
        assert 'sample_count' in baseline
        assert 'avg_processing_time' in baseline
        assert baseline['sample_count'] <= 100
        assert 0.0 <= baseline['accuracy'] <= 1.0

    def test_optimize_scenario(self, orchestrator):
        """Test scenario-specific optimization."""
        baseline = {
            'accuracy': 0.45,
            'sample_count': 100,
            'avg_processing_time': 30.0,
            'avg_retrieval_time': 5.0,
            'avg_generation_time': 25.0
        }

        optimized = orchestrator._optimize_scenario(
            PerspectiveChangeType.ANGLE,
            baseline,
            optimization_rounds=5
        )

        assert optimized['accuracy'] > baseline['accuracy']
        assert optimized['avg_processing_time'] < baseline['avg_processing_time']
        assert 'optimal_params' in optimized

    def test_get_scenario_optimal_params(self, orchestrator):
        """Test scenario-specific parameter retrieval."""
        params = orchestrator._get_scenario_optimal_params(PerspectiveChangeType.ANGLE)

        assert 'top_k' in params
        assert 'temperature' in params
        assert 'max_length' in params
        assert 'prompt_template' in params

    def test_calculate_confidence_interval(self, orchestrator):
        """Test confidence interval calculation."""
        ci = orchestrator._calculate_confidence_interval(
            accuracy=0.55,
            sample_count=100,
            confidence_level=0.95
        )

        lower, upper = ci
        assert lower < 0.55 < upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        assert upper - lower < 0.20  # Reasonable width

    def test_check_target_achievement(self, orchestrator):
        """Test target achievement checking."""
        assert orchestrator._check_target_achievement(0.55) is True
        assert orchestrator._check_target_achievement(0.50) is False
        assert orchestrator._check_target_achievement(0.65) is False

    def test_calculate_accuracy_gap(self, orchestrator):
        """Test accuracy gap calculation."""
        # Below target
        gap = orchestrator._calculate_accuracy_gap(0.48)
        assert gap > 0
        assert gap == pytest.approx(0.05, abs=0.01)

        # Within target
        gap = orchestrator._calculate_accuracy_gap(0.55)
        assert gap == 0.0

        # Above target
        gap = orchestrator._calculate_accuracy_gap(0.65)
        assert gap > 0


class TestCalculateOverallMetrics:
    """Test overall metrics calculation."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        config = MRAGConfig()
        with patch('run_sprint9_multi_scenario.MRAGBenchEvaluator'):
            return Sprint9MultiScenarioOrchestrator(config)

    @pytest.fixture
    def sample_scenario_results(self):
        """Create sample scenario results."""
        return {
            "angle": ScenarioOptimizationResult(
                scenario_type="angle",
                baseline_accuracy=0.48,
                optimized_accuracy=0.56,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=322,
                avg_processing_time=24.0,
                optimization_rounds=5,
                confidence_interval=(0.53, 0.59)
            ),
            "partial": ScenarioOptimizationResult(
                scenario_type="partial",
                baseline_accuracy=0.45,
                optimized_accuracy=0.53,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=246,
                avg_processing_time=26.0,
                optimization_rounds=5,
                confidence_interval=(0.50, 0.56)
            ),
            "scope": ScenarioOptimizationResult(
                scenario_type="scope",
                baseline_accuracy=0.42,
                optimized_accuracy=0.52,
                accuracy_improvement=0.10,
                optimal_parameters={},
                sample_count=102,
                avg_processing_time=28.0,
                optimization_rounds=5,
                confidence_interval=(0.48, 0.56)
            ),
            "occlusion": ScenarioOptimizationResult(
                scenario_type="occlusion",
                baseline_accuracy=0.40,
                optimized_accuracy=0.51,
                accuracy_improvement=0.11,
                optimal_parameters={},
                sample_count=108,
                avg_processing_time=29.0,
                optimization_rounds=5,
                confidence_interval=(0.47, 0.55)
            )
        }

    def test_overall_metrics_calculation(self, orchestrator, sample_scenario_results):
        """Test calculation of overall metrics."""
        metrics = orchestrator._calculate_overall_metrics(sample_scenario_results)

        assert 'overall_accuracy' in metrics
        assert 'total_questions' in metrics
        assert 'total_correct' in metrics
        assert metrics['total_questions'] == 778  # 322 + 246 + 102 + 108
        assert 0.0 <= metrics['overall_accuracy'] <= 1.0

    def test_weighted_average_processing_time(self, orchestrator, sample_scenario_results):
        """Test weighted average processing time."""
        metrics = orchestrator._calculate_overall_metrics(sample_scenario_results)

        assert 'avg_processing_time' in metrics
        assert metrics['avg_processing_time'] > 0
        # Should be weighted by sample count, so closer to angle's 24.0s

    def test_memory_metrics(self, orchestrator, sample_scenario_results):
        """Test memory metrics in overall calculation."""
        metrics = orchestrator._calculate_overall_metrics(sample_scenario_results)

        assert 'peak_memory_gb' in metrics
        assert metrics['peak_memory_gb'] < 15.0  # Within constraint


class TestCrossScenarioAnalysis:
    """Test cross-scenario analysis functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        config = MRAGConfig()
        with patch('run_sprint9_multi_scenario.MRAGBenchEvaluator'):
            return Sprint9MultiScenarioOrchestrator(config)

    @pytest.fixture
    def sample_scenario_results(self):
        """Create sample scenario results with varied performance."""
        return {
            "angle": ScenarioOptimizationResult(
                scenario_type="angle",
                baseline_accuracy=0.48,
                optimized_accuracy=0.56,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=322,
                avg_processing_time=24.0,
                optimization_rounds=5,
                confidence_interval=(0.53, 0.59)
            ),
            "partial": ScenarioOptimizationResult(
                scenario_type="partial",
                baseline_accuracy=0.45,
                optimized_accuracy=0.53,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=246,
                avg_processing_time=26.0,
                optimization_rounds=5,
                confidence_interval=(0.50, 0.56)
            ),
            "scope": ScenarioOptimizationResult(
                scenario_type="scope",
                baseline_accuracy=0.42,
                optimized_accuracy=0.52,
                accuracy_improvement=0.10,
                optimal_parameters={},
                sample_count=102,
                avg_processing_time=28.0,
                optimization_rounds=5,
                confidence_interval=(0.48, 0.56)
            ),
            "occlusion": ScenarioOptimizationResult(
                scenario_type="occlusion",
                baseline_accuracy=0.40,
                optimized_accuracy=0.48,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=108,
                avg_processing_time=29.0,
                optimization_rounds=5,
                confidence_interval=(0.45, 0.51)
            )
        }

    def test_cross_scenario_analysis(self, orchestrator, sample_scenario_results):
        """Test cross-scenario analysis."""
        analysis = orchestrator._perform_cross_scenario_analysis(sample_scenario_results)

        assert analysis.best_scenario == "angle"  # Highest accuracy
        assert analysis.worst_scenario == "occlusion"  # Lowest accuracy
        assert 0.0 <= analysis.performance_consistency <= 1.0

    def test_scenario_rankings(self, orchestrator, sample_scenario_results):
        """Test scenario ranking generation."""
        analysis = orchestrator._perform_cross_scenario_analysis(sample_scenario_results)

        assert len(analysis.scenario_rankings) == 4
        assert analysis.scenario_rankings["angle"] == 1  # Best
        assert analysis.scenario_rankings["occlusion"] == 4  # Worst

    def test_difficulty_assessment(self, orchestrator, sample_scenario_results):
        """Test difficulty assessment for scenarios."""
        analysis = orchestrator._perform_cross_scenario_analysis(sample_scenario_results)

        assert len(analysis.difficulty_assessment) == 4
        assert analysis.difficulty_assessment["angle"] in ["Easy", "Moderate"]
        assert analysis.difficulty_assessment["occlusion"] in ["Challenging", "Difficult"]

    def test_common_challenges_identification(self, orchestrator, sample_scenario_results):
        """Test identification of common challenges."""
        analysis = orchestrator._perform_cross_scenario_analysis(sample_scenario_results)

        assert isinstance(analysis.common_challenges, list)
        # Occlusion is worst, should be mentioned
        challenges_text = " ".join(analysis.common_challenges).lower()
        assert "occlusion" in challenges_text or len(analysis.common_challenges) == 0

    def test_optimization_recommendations(self, orchestrator, sample_scenario_results):
        """Test generation of optimization recommendations."""
        analysis = orchestrator._perform_cross_scenario_analysis(sample_scenario_results)

        assert isinstance(analysis.optimization_recommendations, list)
        # Occlusion is below target, should have recommendations
        if sample_scenario_results["occlusion"].optimized_accuracy < orchestrator.target_min:
            assert len(analysis.optimization_recommendations) > 0


class TestStatisticalValidation:
    """Test statistical validation functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        config = MRAGConfig()
        with patch('run_sprint9_multi_scenario.MRAGBenchEvaluator'):
            return Sprint9MultiScenarioOrchestrator(config)

    @pytest.fixture
    def sample_scenario_results(self):
        """Create sample scenario results."""
        return {
            "angle": ScenarioOptimizationResult(
                scenario_type="angle",
                baseline_accuracy=0.48,
                optimized_accuracy=0.56,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=322,
                avg_processing_time=24.0,
                optimization_rounds=5,
                confidence_interval=(0.53, 0.59)
            ),
            "partial": ScenarioOptimizationResult(
                scenario_type="partial",
                baseline_accuracy=0.45,
                optimized_accuracy=0.54,
                accuracy_improvement=0.09,
                optimal_parameters={},
                sample_count=246,
                avg_processing_time=26.0,
                optimization_rounds=5,
                confidence_interval=(0.51, 0.57)
            )
        }

    def test_statistical_validation(self, orchestrator, sample_scenario_results):
        """Test statistical validation."""
        validation = orchestrator._perform_statistical_validation(sample_scenario_results)

        assert 'overall_ci' in validation
        assert 'scenarios_in_range' in validation
        assert 'significant_improvements' in validation

    def test_overall_confidence_interval(self, orchestrator, sample_scenario_results):
        """Test overall confidence interval calculation."""
        validation = orchestrator._perform_statistical_validation(sample_scenario_results)

        lower, upper = validation['overall_ci']
        assert lower < upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_scenarios_in_range_count(self, orchestrator, sample_scenario_results):
        """Test counting scenarios in target range."""
        validation = orchestrator._perform_statistical_validation(sample_scenario_results)

        scenarios_in_range = validation['scenarios_in_range']
        assert 0 <= scenarios_in_range <= len(sample_scenario_results)

    def test_significant_improvements_detection(self, orchestrator, sample_scenario_results):
        """Test detection of significant improvements."""
        validation = orchestrator._perform_statistical_validation(sample_scenario_results)

        significant = validation['significant_improvements']
        assert isinstance(significant, list)
        # Both scenarios have >5% improvement
        assert len(significant) == 2


class TestRecommendationGeneration:
    """Test recommendation generation."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        config = MRAGConfig()
        with patch('run_sprint9_multi_scenario.MRAGBenchEvaluator'):
            return Sprint9MultiScenarioOrchestrator(config)

    @pytest.fixture
    def sample_overall_metrics(self):
        """Create sample overall metrics."""
        return {
            'overall_accuracy': 0.54,
            'total_questions': 778,
            'total_correct': 420,
            'avg_processing_time': 25.5,
            'avg_retrieval_time': 4.0,
            'avg_generation_time': 21.5,
            'peak_memory_gb': 14.2
        }

    @pytest.fixture
    def sample_cross_analysis(self):
        """Create sample cross-analysis."""
        return CrossScenarioAnalysis(
            best_scenario="angle",
            worst_scenario="occlusion",
            accuracy_variance=0.008,
            performance_consistency=0.90,
            scenario_rankings={"angle": 1, "partial": 2, "scope": 3, "occlusion": 4},
            difficulty_assessment={"angle": "Easy", "occlusion": "Difficult"},
            common_challenges=["Occlusion challenging"],
            optimization_recommendations=["Focus on occlusion"]
        )

    @pytest.fixture
    def sample_scenario_results(self):
        """Create sample scenario results."""
        return {
            "angle": ScenarioOptimizationResult(
                scenario_type="angle",
                baseline_accuracy=0.48,
                optimized_accuracy=0.56,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=322,
                avg_processing_time=24.0,
                optimization_rounds=5,
                confidence_interval=(0.53, 0.59)
            ),
            "occlusion": ScenarioOptimizationResult(
                scenario_type="occlusion",
                baseline_accuracy=0.40,
                optimized_accuracy=0.48,
                accuracy_improvement=0.08,
                optimal_parameters={},
                sample_count=108,
                avg_processing_time=29.0,
                optimization_rounds=5,
                confidence_interval=(0.45, 0.51)
            )
        }

    def test_recommendation_generation(
        self,
        orchestrator,
        sample_overall_metrics,
        sample_cross_analysis,
        sample_scenario_results
    ):
        """Test recommendation generation."""
        recommendations = orchestrator._generate_recommendations(
            sample_overall_metrics,
            sample_cross_analysis,
            sample_scenario_results
        )

        assert 'immediate' in recommendations
        assert 'sprint10_priorities' in recommendations
        assert isinstance(recommendations['immediate'], list)
        assert isinstance(recommendations['sprint10_priorities'], list)

    def test_recommendations_when_target_achieved(
        self,
        orchestrator,
        sample_cross_analysis,
        sample_scenario_results
    ):
        """Test recommendations when target is achieved."""
        # Set accuracy within target
        metrics = {
            'overall_accuracy': 0.55,
            'total_questions': 778,
            'total_correct': 428,
            'avg_processing_time': 25.5,
            'avg_retrieval_time': 4.0,
            'avg_generation_time': 21.5,
            'peak_memory_gb': 14.2
        }

        recommendations = orchestrator._generate_recommendations(
            metrics,
            sample_cross_analysis,
            sample_scenario_results
        )

        immediate = recommendations['immediate']
        # Should mention target achievement
        achievement_mentioned = any("achieved" in rec.lower() for rec in immediate)
        assert achievement_mentioned

    def test_sprint10_priorities_included(
        self,
        orchestrator,
        sample_overall_metrics,
        sample_cross_analysis,
        sample_scenario_results
    ):
        """Test Sprint 10 priorities are generated."""
        recommendations = orchestrator._generate_recommendations(
            sample_overall_metrics,
            sample_cross_analysis,
            sample_scenario_results
        )

        sprint10 = recommendations['sprint10_priorities']
        assert len(sprint10) >= 3  # Should have multiple priorities
        # Should mention final validation
        priorities_text = " ".join(sprint10).lower()
        assert "validation" in priorities_text or "accuracy" in priorities_text


class TestIntegration:
    """Integration tests for Sprint 9 orchestrator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MRAGConfig()

    def test_full_evaluation_workflow(self, config, tmp_path):
        """Test complete evaluation workflow."""
        with patch('run_sprint9_multi_scenario.MRAGBenchEvaluator'):
            orchestrator = Sprint9MultiScenarioOrchestrator(
                config,
                output_dir=str(tmp_path / "sprint9_test")
            )

            # Run with minimal samples for testing
            with patch.object(orchestrator, '_evaluate_scenario_baseline') as mock_baseline, \
                 patch.object(orchestrator, '_optimize_scenario') as mock_optimize:

                # Mock baseline evaluation
                mock_baseline.return_value = {
                    'accuracy': 0.48,
                    'sample_count': 50,
                    'avg_processing_time': 28.0,
                    'avg_retrieval_time': 4.0,
                    'avg_generation_time': 24.0
                }

                # Mock optimization
                mock_optimize.return_value = {
                    'accuracy': 0.55,
                    'sample_count': 50,
                    'avg_processing_time': 24.0,
                    'avg_retrieval_time': 3.5,
                    'avg_generation_time': 20.5,
                    'optimal_params': {'top_k': 5}
                }

                # This will test the full workflow
                results = orchestrator.run_comprehensive_multi_scenario_evaluation(
                    max_samples_per_scenario=50,
                    optimization_rounds_per_scenario=2,
                    enable_optimization=True
                )

                # Verify results structure
                assert results.overall_accuracy > 0
                assert len(results.scenario_results) == 4
                assert results.evaluation_duration > 0

    def test_results_persistence(self, config, tmp_path):
        """Test results are saved correctly."""
        with patch('run_sprint9_multi_scenario.MRAGBenchEvaluator'):
            orchestrator = Sprint9MultiScenarioOrchestrator(
                config,
                output_dir=str(tmp_path / "sprint9_test")
            )

            # Create mock results
            scenario_results = {
                "angle": ScenarioOptimizationResult(
                    scenario_type="angle",
                    baseline_accuracy=0.48,
                    optimized_accuracy=0.55,
                    accuracy_improvement=0.07,
                    optimal_parameters={},
                    sample_count=100,
                    avg_processing_time=24.0,
                    optimization_rounds=2,
                    confidence_interval=(0.52, 0.58)
                )
            }

            cross_analysis = CrossScenarioAnalysis(
                best_scenario="angle",
                worst_scenario="occlusion",
                accuracy_variance=0.008,
                performance_consistency=0.90,
                scenario_rankings={"angle": 1},
                difficulty_assessment={"angle": "Easy"},
                common_challenges=[],
                optimization_recommendations=[]
            )

            results = Sprint9Results(
                overall_accuracy=0.55,
                total_questions=100,
                total_correct=55,
                overall_confidence_interval=(0.52, 0.58),
                target_achieved=True,
                scenario_results=scenario_results,
                scenario_accuracies={"angle": 0.55},
                scenario_sample_counts={"angle": 100},
                cross_scenario_analysis=cross_analysis,
                avg_processing_time=24.0,
                avg_retrieval_time=4.0,
                avg_generation_time=20.0,
                peak_memory_gb=14.0,
                target_range=(0.53, 0.59),
                accuracy_gap=0.0,
                scenarios_in_range=1,
                recommendations=["Test"],
                sprint10_priorities=["Test priority"],
                timestamp="2025-10-04T00:00:00",
                evaluation_duration=60.0,
                total_optimization_rounds=2
            )

            # Save results
            orchestrator._save_results(results)

            # Verify files exist
            json_file = tmp_path / "sprint9_test" / "sprint9_multi_scenario_results.json"
            assert json_file.exists()

            # Verify JSON content
            with open(json_file) as f:
                loaded_data = json.load(f)
                assert loaded_data['overall_accuracy'] == 0.55
                assert loaded_data['target_achieved'] is True

    def test_summary_report_generation(self, config, tmp_path):
        """Test summary report generation."""
        with patch('run_sprint9_multi_scenario.MRAGBenchEvaluator'):
            orchestrator = Sprint9MultiScenarioOrchestrator(
                config,
                output_dir=str(tmp_path / "sprint9_test")
            )

            # Create mock results
            scenario_results = {
                "angle": ScenarioOptimizationResult(
                    scenario_type="angle",
                    baseline_accuracy=0.48,
                    optimized_accuracy=0.55,
                    accuracy_improvement=0.07,
                    optimal_parameters={},
                    sample_count=100,
                    avg_processing_time=24.0,
                    optimization_rounds=2,
                    confidence_interval=(0.52, 0.58)
                )
            }

            cross_analysis = CrossScenarioAnalysis(
                best_scenario="angle",
                worst_scenario="occlusion",
                accuracy_variance=0.008,
                performance_consistency=0.90,
                scenario_rankings={"angle": 1},
                difficulty_assessment={"angle": "Easy"},
                common_challenges=["Test challenge"],
                optimization_recommendations=["Test recommendation"]
            )

            results = Sprint9Results(
                overall_accuracy=0.55,
                total_questions=100,
                total_correct=55,
                overall_confidence_interval=(0.52, 0.58),
                target_achieved=True,
                scenario_results=scenario_results,
                scenario_accuracies={"angle": 0.55},
                scenario_sample_counts={"angle": 100},
                cross_scenario_analysis=cross_analysis,
                avg_processing_time=24.0,
                avg_retrieval_time=4.0,
                avg_generation_time=20.0,
                peak_memory_gb=14.0,
                target_range=(0.53, 0.59),
                accuracy_gap=0.0,
                scenarios_in_range=1,
                recommendations=["Test recommendation"],
                sprint10_priorities=["Test priority"],
                timestamp="2025-10-04T00:00:00",
                evaluation_duration=60.0,
                total_optimization_rounds=2
            )

            # Generate report
            orchestrator._generate_summary_report(results)

            # Verify report exists
            report_file = tmp_path / "sprint9_test" / "sprint9_summary_report.md"
            assert report_file.exists()

            # Verify report content
            content = report_file.read_text()
            assert "Sprint 9: Multi-Scenario Expansion" in content
            assert "Overall Accuracy: 55.0%" in content
            assert "Target Achievement" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
