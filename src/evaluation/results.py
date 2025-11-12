"""
Evaluation Results

Data structures for storing and analyzing MRAG-Bench evaluation results.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import time


@dataclass
class ScenarioResults:
    """Results for a single perspective change scenario."""
    scenario_type: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_processing_time: float
    memory_usage: Dict[str, float]
    failed_questions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.total_questions > 0:
            self.accuracy = self.correct_answers / self.total_questions
        else:
            self.accuracy = 0.0


@dataclass
class EvaluationResults:
    """Complete MRAG-Bench evaluation results."""
    overall_accuracy: float
    scenario_results: Dict[str, ScenarioResults]
    total_questions: int
    total_correct: int
    evaluation_time: float
    system_performance: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def __post_init__(self):
        if self.total_questions > 0:
            self.overall_accuracy = self.total_correct / self.total_questions
        else:
            self.overall_accuracy = 0.0

    def get_accuracy_by_scenario(self) -> Dict[str, float]:
        """Get accuracy for each scenario type."""
        return {
            scenario: results.accuracy
            for scenario, results in self.scenario_results.items()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        return {
            "overall_accuracy": self.overall_accuracy,
            "scenario_accuracies": self.get_accuracy_by_scenario(),
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "evaluation_time": self.evaluation_time,
            "avg_processing_time": sum(
                results.avg_processing_time
                for results in self.scenario_results.values()
            ) / len(self.scenario_results) if self.scenario_results else 0.0,
            "memory_stats": self.system_performance.get("memory", {}),
            "timestamp": self.timestamp
        }

    def meets_target_accuracy(self, min_accuracy: float = 0.53, max_accuracy: float = 0.59) -> bool:
        """
        Check if results meet MRAG-Bench target accuracy range.

        Args:
            min_accuracy: Minimum target accuracy (53%)
            max_accuracy: Maximum target accuracy (59%)

        Returns:
            True if accuracy is within target range
        """
        return min_accuracy <= self.overall_accuracy <= max_accuracy

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "overall_accuracy": self.overall_accuracy,
            "scenario_results": {
                scenario: {
                    "scenario_type": results.scenario_type,
                    "total_questions": results.total_questions,
                    "correct_answers": results.correct_answers,
                    "accuracy": results.accuracy,
                    "avg_processing_time": results.avg_processing_time,
                    "memory_usage": results.memory_usage,
                    "failed_questions": results.failed_questions
                }
                for scenario, results in self.scenario_results.items()
            },
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "evaluation_time": self.evaluation_time,
            "system_performance": self.system_performance,
            "timestamp": self.timestamp
        }

    def save_to_file(self, filepath: str) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'EvaluationResults':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct ScenarioResults objects
        scenario_results = {}
        for scenario, results_data in data["scenario_results"].items():
            scenario_results[scenario] = ScenarioResults(**results_data)

        # Create EvaluationResults object
        return cls(
            overall_accuracy=data["overall_accuracy"],
            scenario_results=scenario_results,
            total_questions=data["total_questions"],
            total_correct=data["total_correct"],
            evaluation_time=data["evaluation_time"],
            system_performance=data["system_performance"],
            timestamp=data["timestamp"]
        )