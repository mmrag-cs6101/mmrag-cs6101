"""
Performance Optimizer for MRAG-Bench System

Implements intelligent hyperparameter optimization and performance tuning to achieve
target accuracy (53-59%) while maintaining memory constraints and processing speed.
"""

import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import itertools
from scipy.stats import uniform, randint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from ..utils.error_handling import handle_errors, MRAGError

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    GRID_SEARCH = "grid"
    RANDOM_SEARCH = "random"
    BAYESIAN = "bayesian"
    ADAPTIVE = "adaptive"


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    target_accuracy_range: Tuple[float, float] = (0.53, 0.59)
    max_processing_time: float = 30.0
    memory_limit_gb: float = 15.0
    optimization_strategy: str = "bayesian"
    max_iterations: int = 50
    exploration_factor: float = 0.3
    convergence_threshold: float = 0.01
    search_space_config: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    best_config: Dict[str, Any]
    best_accuracy: float
    optimization_iterations: int
    target_achieved: bool
    improvement_over_baseline: float
    convergence_history: List[float]


class PerformanceOptimizer:
    """
    Intelligent performance optimizer for MRAG-Bench system.

    Features:
    - Multiple optimization strategies (Grid, Random, Bayesian, Adaptive)
    - Smart parameter space exploration with medical domain knowledge
    - Memory-aware optimization with constraint handling
    - Performance-accuracy trade-off optimization
    - Convergence detection and early stopping
    - Domain-specific parameter importance ranking
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize performance optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.optimization_history = []
        self.best_config = None
        self.best_accuracy = 0.0

        # Initialize search space
        self.search_space = self._initialize_search_space()

        # Initialize optimization strategy
        self.strategy = OptimizationStrategy(config.optimization_strategy)

        # Initialize Bayesian optimization components
        if self.strategy == OptimizationStrategy.BAYESIAN:
            self.gp_regressor = GaussianProcessRegressor(
                kernel=Matern(length_scale=1.0, nu=1.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
            self.X_observed = []
            self.y_observed = []

        logger.info(f"Performance optimizer initialized with {config.optimization_strategy} strategy")

    def _initialize_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Initialize optimization search space with domain knowledge."""
        if self.config.search_space_config:
            return self.config.search_space_config

        # Default search space optimized for medical image QA
        return {
            "retrieval": {
                "top_k": {
                    "type": "discrete",
                    "values": [3, 5, 7, 10, 15],
                    "default": 5,
                    "importance": 0.9  # High importance for medical accuracy
                },
                "similarity_threshold": {
                    "type": "continuous",
                    "range": (0.5, 0.95),
                    "default": 0.7,
                    "importance": 0.7
                },
                "batch_size": {
                    "type": "discrete",
                    "values": [4, 8, 16, 32],
                    "default": 8,
                    "importance": 0.3  # Mainly affects speed
                }
            },
            "generation": {
                "temperature": {
                    "type": "continuous",
                    "range": (0.1, 1.0),
                    "default": 0.3,
                    "importance": 0.8  # Critical for medical consistency
                },
                "top_p": {
                    "type": "continuous",
                    "range": (0.7, 0.99),
                    "default": 0.9,
                    "importance": 0.6
                },
                "max_length": {
                    "type": "discrete",
                    "values": [128, 256, 512, 1024],
                    "default": 512,
                    "importance": 0.4  # Affects speed more than accuracy
                },
                "do_sample": {
                    "type": "categorical",
                    "values": [True, False],
                    "default": True,
                    "importance": 0.5
                }
            },
            "memory": {
                "sequential_loading": {
                    "type": "categorical",
                    "values": [True, False],
                    "default": True,
                    "importance": 0.6  # Important for memory efficiency
                },
                "aggressive_cleanup": {
                    "type": "categorical",
                    "values": [True, False],
                    "default": True,
                    "importance": 0.4
                }
            }
        }

    @handle_errors
    def suggest_configurations(
        self,
        current_accuracy: float,
        num_suggestions: int = 3,
        exploration_factor: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Suggest optimal configurations based on current performance.

        Args:
            current_accuracy: Current system accuracy
            num_suggestions: Number of configurations to suggest
            exploration_factor: Balance between exploration and exploitation

        Returns:
            List of suggested configurations
        """
        logger.debug(f"Suggesting {num_suggestions} configurations (exploration: {exploration_factor:.2f})")

        if self.strategy == OptimizationStrategy.GRID_SEARCH:
            return self._grid_search_suggestions(num_suggestions)
        elif self.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return self._random_search_suggestions(num_suggestions)
        elif self.strategy == OptimizationStrategy.BAYESIAN:
            return self._bayesian_suggestions(num_suggestions, exploration_factor)
        elif self.strategy == OptimizationStrategy.ADAPTIVE:
            return self._adaptive_suggestions(current_accuracy, num_suggestions, exploration_factor)
        else:
            raise ValueError(f"Unknown optimization strategy: {self.strategy}")

    def _grid_search_suggestions(self, num_suggestions: int) -> List[Dict[str, Any]]:
        """Generate grid search configurations."""
        configs = []

        # Create parameter grids for most important parameters
        important_params = self._get_important_parameters()

        # Generate grid combinations
        param_combinations = []
        for category, params in important_params.items():
            for param_name, param_config in params.items():
                if param_config["type"] == "discrete":
                    param_combinations.append(
                        [(category, param_name, value) for value in param_config["values"]]
                    )
                elif param_config["type"] == "continuous":
                    # Discretize continuous parameters
                    min_val, max_val = param_config["range"]
                    values = np.linspace(min_val, max_val, 5).tolist()
                    param_combinations.append(
                        [(category, param_name, value) for value in values]
                    )

        # Generate configurations from combinations
        for combination in itertools.product(*param_combinations[:4]):  # Limit to avoid explosion
            config = {"retrieval": {}, "generation": {}, "memory": {}}
            for category, param_name, value in combination:
                config[category][param_name] = value
            configs.append(config)

        # Return requested number of suggestions
        return configs[:num_suggestions]

    def _random_search_suggestions(self, num_suggestions: int) -> List[Dict[str, Any]]:
        """Generate random search configurations."""
        configs = []

        for _ in range(num_suggestions):
            config = {"retrieval": {}, "generation": {}, "memory": {}}

            for category, params in self.search_space.items():
                for param_name, param_config in params.items():
                    if param_config["type"] == "discrete":
                        config[category][param_name] = random.choice(param_config["values"])
                    elif param_config["type"] == "continuous":
                        min_val, max_val = param_config["range"]
                        config[category][param_name] = random.uniform(min_val, max_val)
                    elif param_config["type"] == "categorical":
                        config[category][param_name] = random.choice(param_config["values"])

            configs.append(config)

        return configs

    def _bayesian_suggestions(self, num_suggestions: int, exploration_factor: float) -> List[Dict[str, Any]]:
        """Generate Bayesian optimization suggestions."""
        if len(self.y_observed) < 3:
            # Not enough data for GP, use random search
            return self._random_search_suggestions(num_suggestions)

        configs = []

        # Fit Gaussian Process
        X_array = np.array(self.X_observed)
        y_array = np.array(self.y_observed)
        self.gp_regressor.fit(X_array, y_array)

        # Generate candidates and select best using acquisition function
        candidates = self._generate_candidate_points(num_suggestions * 10)

        best_candidates = []
        for candidate in candidates:
            # Calculate acquisition function (Upper Confidence Bound)
            x_point = self._config_to_vector(candidate)
            if len(x_point) == X_array.shape[1]:  # Ensure consistent dimensionality
                mean, std = self.gp_regressor.predict([x_point], return_std=True)
                ucb = mean[0] + exploration_factor * std[0]
                best_candidates.append((candidate, ucb))

        # Sort by acquisition function value and return top suggestions
        best_candidates.sort(key=lambda x: x[1], reverse=True)
        configs = [config for config, _ in best_candidates[:num_suggestions]]

        if len(configs) < num_suggestions:
            # Fill remaining with random suggestions
            configs.extend(self._random_search_suggestions(num_suggestions - len(configs)))

        return configs

    def _adaptive_suggestions(
        self,
        current_accuracy: float,
        num_suggestions: int,
        exploration_factor: float
    ) -> List[Dict[str, Any]]:
        """Generate adaptive suggestions based on current performance."""
        configs = []

        # Determine focus based on current accuracy
        if current_accuracy < self.config.target_accuracy_range[0]:
            # Below target: focus on accuracy improvement
            configs.extend(self._accuracy_focused_suggestions(num_suggestions // 2))
            configs.extend(self._random_search_suggestions(num_suggestions - len(configs)))
        elif current_accuracy > self.config.target_accuracy_range[1]:
            # Above target: focus on speed optimization
            configs.extend(self._speed_focused_suggestions(num_suggestions // 2))
            configs.extend(self._random_search_suggestions(num_suggestions - len(configs)))
        else:
            # In target range: balanced exploration
            configs.extend(self._balanced_suggestions(num_suggestions))

        return configs[:num_suggestions]

    def _accuracy_focused_suggestions(self, num_suggestions: int) -> List[Dict[str, Any]]:
        """Generate configurations focused on improving accuracy."""
        configs = []

        # Parameters that typically improve accuracy in medical VQA
        accuracy_configs = [
            {
                "retrieval": {"top_k": 10, "similarity_threshold": 0.6},
                "generation": {"temperature": 0.1, "top_p": 0.9, "max_length": 512}
            },
            {
                "retrieval": {"top_k": 15, "similarity_threshold": 0.5},
                "generation": {"temperature": 0.2, "top_p": 0.95, "max_length": 1024}
            },
            {
                "retrieval": {"top_k": 7, "similarity_threshold": 0.7},
                "generation": {"temperature": 0.3, "top_p": 0.85, "max_length": 512}
            }
        ]

        return accuracy_configs[:num_suggestions]

    def _speed_focused_suggestions(self, num_suggestions: int) -> List[Dict[str, Any]]:
        """Generate configurations focused on improving speed."""
        configs = []

        # Parameters that typically improve speed
        speed_configs = [
            {
                "retrieval": {"top_k": 3, "batch_size": 16},
                "generation": {"max_length": 256, "temperature": 0.5},
                "memory": {"sequential_loading": True, "aggressive_cleanup": True}
            },
            {
                "retrieval": {"top_k": 5, "batch_size": 32},
                "generation": {"max_length": 128, "temperature": 0.7},
                "memory": {"sequential_loading": True, "aggressive_cleanup": True}
            }
        ]

        return speed_configs[:num_suggestions]

    def _balanced_suggestions(self, num_suggestions: int) -> List[Dict[str, Any]]:
        """Generate balanced suggestions for accuracy-speed trade-off."""
        configs = []

        # Use combination of strategies
        configs.extend(self._random_search_suggestions(num_suggestions // 2))
        if len(self.y_observed) >= 3:
            configs.extend(self._bayesian_suggestions(num_suggestions - len(configs), 0.2))
        else:
            configs.extend(self._random_search_suggestions(num_suggestions - len(configs)))

        return configs

    def _generate_candidate_points(self, num_candidates: int) -> List[Dict[str, Any]]:
        """Generate candidate configuration points for Bayesian optimization."""
        candidates = []

        for _ in range(num_candidates):
            config = {"retrieval": {}, "generation": {}, "memory": {}}

            for category, params in self.search_space.items():
                for param_name, param_config in params.items():
                    if param_config["type"] == "discrete":
                        config[category][param_name] = random.choice(param_config["values"])
                    elif param_config["type"] == "continuous":
                        min_val, max_val = param_config["range"]
                        config[category][param_name] = random.uniform(min_val, max_val)
                    elif param_config["type"] == "categorical":
                        config[category][param_name] = random.choice(param_config["values"])

            candidates.append(config)

        return candidates

    def _config_to_vector(self, config: Dict[str, Any]) -> List[float]:
        """Convert configuration to numerical vector for GP."""
        vector = []

        for category in ["retrieval", "generation", "memory"]:
            if category in config:
                for param_name in sorted(config[category].keys()):
                    value = config[category][param_name]

                    # Convert to numerical value
                    if isinstance(value, bool):
                        vector.append(1.0 if value else 0.0)
                    elif isinstance(value, (int, float)):
                        vector.append(float(value))
                    else:
                        # String values - use hash for consistency
                        vector.append(float(hash(str(value)) % 100) / 100.0)

        return vector

    def _get_important_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get most important parameters for optimization."""
        important_params = {}

        for category, params in self.search_space.items():
            important_params[category] = {}
            for param_name, param_config in params.items():
                if param_config.get("importance", 0.5) > 0.6:
                    important_params[category][param_name] = param_config

        return important_params

    def update_observation(self, config: Dict[str, Any], accuracy: float) -> None:
        """Update optimizer with new observation."""
        self.optimization_history.append({
            "config": config,
            "accuracy": accuracy
        })

        # Update best configuration
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_config = config

        # Update Bayesian optimization data
        if self.strategy == OptimizationStrategy.BAYESIAN:
            x_vector = self._config_to_vector(config)
            self.X_observed.append(x_vector)
            self.y_observed.append(accuracy)

        logger.debug(f"Updated observation: accuracy={accuracy:.3f}, total_observations={len(self.optimization_history)}")

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "strategy": self.strategy.value,
            "observations": len(self.optimization_history),
            "best_accuracy": self.best_accuracy,
            "best_config": self.best_config,
            "target_range": self.config.target_accuracy_range,
            "target_achieved": (
                self.config.target_accuracy_range[0] <= self.best_accuracy <= self.config.target_accuracy_range[1]
            ) if self.best_accuracy > 0 else False
        }

    def suggest_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Suggest parameter bounds based on optimization history."""
        if len(self.optimization_history) < 5:
            return self.search_space

        # Analyze successful configurations
        successful_configs = [
            obs["config"] for obs in self.optimization_history
            if obs["accuracy"] >= self.config.target_accuracy_range[0]
        ]

        if not successful_configs:
            return self.search_space

        bounds = {}
        for category in ["retrieval", "generation", "memory"]:
            bounds[category] = {}
            for param_name in self.search_space.get(category, {}):
                values = []
                for config in successful_configs:
                    if category in config and param_name in config[category]:
                        values.append(config[category][param_name])

                if values:
                    if isinstance(values[0], (int, float)):
                        bounds[category][param_name] = {
                            "min": min(values),
                            "max": max(values),
                            "mean": np.mean(values),
                            "std": np.std(values)
                        }

        return bounds

    def generate_optimization_report(self) -> str:
        """Generate optimization report."""
        if not self.optimization_history:
            return "No optimization data available."

        report = f"""
Performance Optimization Report
==============================

Strategy: {self.strategy.value}
Total Observations: {len(self.optimization_history)}
Best Accuracy: {self.best_accuracy:.1%}
Target Range: {self.config.target_accuracy_range[0]:.1%} - {self.config.target_accuracy_range[1]:.1%}
Target Achieved: {'✓' if self.config.target_accuracy_range[0] <= self.best_accuracy <= self.config.target_accuracy_range[1] else '✗'}

Best Configuration:
{self.best_config}

Accuracy Progress:
{[obs['accuracy'] for obs in self.optimization_history[-10:]]}

Parameter Analysis:
"""

        # Add parameter importance analysis
        bounds = self.suggest_parameter_bounds()
        for category, params in bounds.items():
            report += f"\n{category.upper()}:\n"
            for param_name, stats in params.items():
                if isinstance(stats, dict) and "mean" in stats:
                    report += f"  {param_name}: {stats['mean']:.3f} ± {stats['std']:.3f}\n"

        return report