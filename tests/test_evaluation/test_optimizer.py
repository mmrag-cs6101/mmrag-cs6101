"""
Unit Tests for Performance Optimizer

Tests for hyperparameter optimization and performance tuning functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.evaluation.optimizer import (
    PerformanceOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationStrategy
)


class TestOptimizationConfig:
    """Test cases for OptimizationConfig."""

    def test_default_config(self):
        """Test default optimization configuration."""
        config = OptimizationConfig()

        assert config.target_accuracy_range == (0.53, 0.59)
        assert config.max_processing_time == 30.0
        assert config.memory_limit_gb == 15.0
        assert config.optimization_strategy == "bayesian"
        assert config.max_iterations == 50
        assert config.exploration_factor == 0.3
        assert config.convergence_threshold == 0.01

    def test_custom_config(self):
        """Test custom optimization configuration."""
        config = OptimizationConfig(
            target_accuracy_range=(0.6, 0.7),
            max_processing_time=20.0,
            optimization_strategy="random",
            max_iterations=30
        )

        assert config.target_accuracy_range == (0.6, 0.7)
        assert config.max_processing_time == 20.0
        assert config.optimization_strategy == "random"
        assert config.max_iterations == 30


class TestPerformanceOptimizer:
    """Test cases for PerformanceOptimizer."""

    @pytest.fixture
    def basic_config(self):
        """Create basic optimization configuration."""
        return OptimizationConfig(
            target_accuracy_range=(0.53, 0.59),
            optimization_strategy="random",
            max_iterations=10
        )

    @pytest.fixture
    def bayesian_config(self):
        """Create Bayesian optimization configuration."""
        return OptimizationConfig(
            target_accuracy_range=(0.53, 0.59),
            optimization_strategy="bayesian",
            max_iterations=10
        )

    def test_optimizer_initialization(self, basic_config):
        """Test optimizer initialization."""
        optimizer = PerformanceOptimizer(basic_config)

        assert optimizer.config == basic_config
        assert optimizer.strategy == OptimizationStrategy.RANDOM_SEARCH
        assert len(optimizer.optimization_history) == 0
        assert optimizer.best_accuracy == 0.0
        assert optimizer.best_config is None
        assert optimizer.search_space is not None

    def test_search_space_initialization(self, basic_config):
        """Test search space initialization."""
        optimizer = PerformanceOptimizer(basic_config)

        search_space = optimizer.search_space

        assert "retrieval" in search_space
        assert "generation" in search_space
        assert "memory" in search_space

        # Test retrieval parameters
        assert "top_k" in search_space["retrieval"]
        assert "similarity_threshold" in search_space["retrieval"]

        # Test generation parameters
        assert "temperature" in search_space["generation"]
        assert "top_p" in search_space["generation"]
        assert "max_length" in search_space["generation"]

    def test_custom_search_space(self):
        """Test custom search space configuration."""
        custom_search_space = {
            "retrieval": {
                "top_k": {"type": "discrete", "values": [1, 2, 3], "importance": 0.8}
            },
            "generation": {
                "temperature": {"type": "continuous", "range": (0.0, 1.0), "importance": 0.9}
            }
        }

        config = OptimizationConfig(
            search_space_config=custom_search_space,
            optimization_strategy="random"
        )
        optimizer = PerformanceOptimizer(config)

        assert optimizer.search_space == custom_search_space
        assert optimizer.search_space["retrieval"]["top_k"]["values"] == [1, 2, 3]

    def test_random_search_suggestions(self, basic_config):
        """Test random search configuration suggestions."""
        optimizer = PerformanceOptimizer(basic_config)

        suggestions = optimizer.suggest_configurations(
            current_accuracy=0.5,
            num_suggestions=3
        )

        assert len(suggestions) == 3
        for config in suggestions:
            assert "retrieval" in config
            assert "generation" in config
            assert "memory" in config

            # Validate parameter ranges
            if "top_k" in config["retrieval"]:
                assert config["retrieval"]["top_k"] in optimizer.search_space["retrieval"]["top_k"]["values"]

            if "temperature" in config["generation"]:
                temp_range = optimizer.search_space["generation"]["temperature"]["range"]
                assert temp_range[0] <= config["generation"]["temperature"] <= temp_range[1]

    def test_grid_search_suggestions(self):
        """Test grid search configuration suggestions."""
        config = OptimizationConfig(optimization_strategy="grid")
        optimizer = PerformanceOptimizer(config)

        suggestions = optimizer.suggest_configurations(
            current_accuracy=0.5,
            num_suggestions=2
        )

        assert len(suggestions) <= 2
        # Grid search should produce valid configurations
        for config in suggestions:
            assert isinstance(config, dict)
            assert "retrieval" in config or "generation" in config

    def test_bayesian_optimization_initialization(self, bayesian_config):
        """Test Bayesian optimization initialization."""
        optimizer = PerformanceOptimizer(bayesian_config)

        assert optimizer.strategy == OptimizationStrategy.BAYESIAN
        assert hasattr(optimizer, 'gp_regressor')
        assert len(optimizer.X_observed) == 0
        assert len(optimizer.y_observed) == 0

    def test_bayesian_suggestions_with_data(self, bayesian_config):
        """Test Bayesian suggestions with observation data."""
        optimizer = PerformanceOptimizer(bayesian_config)

        # Add some observations
        for i in range(5):
            config = {
                "retrieval": {"top_k": 5 + i, "similarity_threshold": 0.7},
                "generation": {"temperature": 0.3 + i * 0.1}
            }
            accuracy = 0.5 + i * 0.02
            optimizer.update_observation(config, accuracy)

        suggestions = optimizer.suggest_configurations(
            current_accuracy=0.55,
            num_suggestions=2
        )

        assert len(suggestions) == 2
        # Should return valid configurations
        for config in suggestions:
            assert isinstance(config, dict)

    def test_adaptive_suggestions_below_target(self):
        """Test adaptive suggestions when accuracy is below target."""
        config = OptimizationConfig(
            optimization_strategy="adaptive",
            target_accuracy_range=(0.53, 0.59)
        )
        optimizer = PerformanceOptimizer(config)

        # Test below target accuracy
        suggestions = optimizer._adaptive_suggestions(
            current_accuracy=0.45,  # Below 0.53
            num_suggestions=3,
            exploration_factor=0.3
        )

        assert len(suggestions) == 3
        # Should focus on accuracy improvement

    def test_adaptive_suggestions_above_target(self):
        """Test adaptive suggestions when accuracy is above target."""
        config = OptimizationConfig(
            optimization_strategy="adaptive",
            target_accuracy_range=(0.53, 0.59)
        )
        optimizer = PerformanceOptimizer(config)

        # Test above target accuracy
        suggestions = optimizer._adaptive_suggestions(
            current_accuracy=0.65,  # Above 0.59
            num_suggestions=3,
            exploration_factor=0.3
        )

        assert len(suggestions) == 3
        # Should focus on speed optimization

    def test_update_observation(self, basic_config):
        """Test updating optimizer with new observations."""
        optimizer = PerformanceOptimizer(basic_config)

        config1 = {"retrieval": {"top_k": 5}, "generation": {"temperature": 0.3}}
        accuracy1 = 0.55

        optimizer.update_observation(config1, accuracy1)

        assert len(optimizer.optimization_history) == 1
        assert optimizer.best_accuracy == accuracy1
        assert optimizer.best_config == config1

        # Add better observation
        config2 = {"retrieval": {"top_k": 7}, "generation": {"temperature": 0.2}}
        accuracy2 = 0.60

        optimizer.update_observation(config2, accuracy2)

        assert len(optimizer.optimization_history) == 2
        assert optimizer.best_accuracy == accuracy2
        assert optimizer.best_config == config2

    def test_config_to_vector_conversion(self, basic_config):
        """Test configuration to vector conversion for Bayesian optimization."""
        optimizer = PerformanceOptimizer(basic_config)

        config = {
            "retrieval": {"top_k": 5, "similarity_threshold": 0.7},
            "generation": {"temperature": 0.3, "do_sample": True},
            "memory": {"sequential_loading": False}
        }

        vector = optimizer._config_to_vector(config)

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_accuracy_focused_suggestions(self, basic_config):
        """Test accuracy-focused configuration suggestions."""
        optimizer = PerformanceOptimizer(basic_config)

        suggestions = optimizer._accuracy_focused_suggestions(2)

        assert len(suggestions) <= 2
        for config in suggestions:
            # Accuracy-focused configs typically have:
            # - Higher top_k for better retrieval
            # - Lower temperature for more deterministic generation
            if "retrieval" in config and "top_k" in config["retrieval"]:
                assert config["retrieval"]["top_k"] >= 5

            if "generation" in config and "temperature" in config["generation"]:
                assert config["generation"]["temperature"] <= 0.5

    def test_speed_focused_suggestions(self, basic_config):
        """Test speed-focused configuration suggestions."""
        optimizer = PerformanceOptimizer(basic_config)

        suggestions = optimizer._speed_focused_suggestions(2)

        assert len(suggestions) <= 2
        for config in suggestions:
            # Speed-focused configs typically have:
            # - Lower top_k for faster retrieval
            # - Shorter max_length for faster generation
            if "retrieval" in config and "top_k" in config["retrieval"]:
                assert config["retrieval"]["top_k"] <= 5

            if "generation" in config and "max_length" in config["generation"]:
                assert config["generation"]["max_length"] <= 512

    def test_get_optimization_status(self, basic_config):
        """Test optimization status reporting."""
        optimizer = PerformanceOptimizer(basic_config)

        # Initial status
        status = optimizer.get_optimization_status()

        assert status["strategy"] == "random"
        assert status["observations"] == 0
        assert status["best_accuracy"] == 0.0
        assert status["best_config"] is None
        assert status["target_range"] == (0.53, 0.59)
        assert status["target_achieved"] == False

        # Add observation
        config = {"retrieval": {"top_k": 5}}
        optimizer.update_observation(config, 0.55)

        status = optimizer.get_optimization_status()
        assert status["observations"] == 1
        assert status["best_accuracy"] == 0.55
        assert status["target_achieved"] == True  # 0.55 is in range

    def test_parameter_bounds_suggestion(self, basic_config):
        """Test parameter bounds suggestion based on successful configs."""
        optimizer = PerformanceOptimizer(basic_config)

        # Add successful configurations
        successful_configs = [
            {"retrieval": {"top_k": 7, "similarity_threshold": 0.8}, "generation": {"temperature": 0.2}},
            {"retrieval": {"top_k": 5, "similarity_threshold": 0.7}, "generation": {"temperature": 0.3}},
            {"retrieval": {"top_k": 9, "similarity_threshold": 0.75}, "generation": {"temperature": 0.25}}
        ]

        for config in successful_configs:
            optimizer.update_observation(config, 0.55)  # All successful (in target range)

        bounds = optimizer.suggest_parameter_bounds()

        assert isinstance(bounds, dict)
        # Should contain analysis of successful parameter ranges

    def test_generate_optimization_report(self, basic_config):
        """Test optimization report generation."""
        optimizer = PerformanceOptimizer(basic_config)

        # Test with no data
        report = optimizer.generate_optimization_report()
        assert "No optimization data available" in report

        # Add some observations
        configs = [
            {"retrieval": {"top_k": 5}, "generation": {"temperature": 0.3}},
            {"retrieval": {"top_k": 7}, "generation": {"temperature": 0.2}},
            {"retrieval": {"top_k": 3}, "generation": {"temperature": 0.5}}
        ]
        accuracies = [0.52, 0.58, 0.45]

        for config, accuracy in zip(configs, accuracies):
            optimizer.update_observation(config, accuracy)

        report = optimizer.generate_optimization_report()

        assert "Performance Optimization Report" in report
        assert "Strategy: random" in report
        assert "Total Observations: 3" in report
        assert "Best Accuracy: 58.0%" in report
        assert "Target Achieved: ✓" in report

    def test_important_parameters_extraction(self, basic_config):
        """Test extraction of important parameters."""
        optimizer = PerformanceOptimizer(basic_config)

        important_params = optimizer._get_important_parameters()

        assert isinstance(important_params, dict)
        # Should include parameters with importance > 0.6
        for category, params in important_params.items():
            for param_name, param_config in params.items():
                assert param_config.get("importance", 0.5) > 0.6

    def test_candidate_point_generation(self, basic_config):
        """Test candidate point generation for Bayesian optimization."""
        optimizer = PerformanceOptimizer(basic_config)

        candidates = optimizer._generate_candidate_points(5)

        assert len(candidates) == 5
        for candidate in candidates:
            assert isinstance(candidate, dict)
            # Should contain valid parameter values within search space ranges


class TestOptimizationStrategy:
    """Test cases for OptimizationStrategy enum."""

    def test_optimization_strategies(self):
        """Test optimization strategy enumeration."""
        assert OptimizationStrategy.GRID_SEARCH.value == "grid"
        assert OptimizationStrategy.RANDOM_SEARCH.value == "random"
        assert OptimizationStrategy.BAYESIAN.value == "bayesian"
        assert OptimizationStrategy.ADAPTIVE.value == "adaptive"

    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        assert OptimizationStrategy("grid") == OptimizationStrategy.GRID_SEARCH
        assert OptimizationStrategy("random") == OptimizationStrategy.RANDOM_SEARCH
        assert OptimizationStrategy("bayesian") == OptimizationStrategy.BAYESIAN
        assert OptimizationStrategy("adaptive") == OptimizationStrategy.ADAPTIVE


class TestOptimizationResult:
    """Test cases for OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            best_config={"retrieval": {"top_k": 7}},
            best_accuracy=0.58,
            optimization_iterations=15,
            target_achieved=True,
            improvement_over_baseline=0.08,
            convergence_history=[0.50, 0.52, 0.55, 0.58]
        )

        assert result.best_config == {"retrieval": {"top_k": 7}}
        assert result.best_accuracy == 0.58
        assert result.optimization_iterations == 15
        assert result.target_achieved == True
        assert result.improvement_over_baseline == 0.08
        assert len(result.convergence_history) == 4

    def test_optimization_result_validation(self):
        """Test optimization result validation."""
        result = OptimizationResult(
            best_config={},
            best_accuracy=0.55,
            optimization_iterations=10,
            target_achieved=True,
            improvement_over_baseline=0.05,
            convergence_history=[]
        )

        # Validate accuracy is reasonable
        assert 0.0 <= result.best_accuracy <= 1.0

        # Validate improvement is consistent with target achievement
        if result.target_achieved and result.improvement_over_baseline > 0:
            assert result.best_accuracy > 0.5  # Should be reasonably good

        # Validate iterations
        assert result.optimization_iterations >= 0