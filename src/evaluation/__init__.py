"""
Evaluation Module

Provides MRAG-Bench evaluation framework for accuracy calculation and performance analysis.
Implements evaluation methodology matching the original paper.
"""

from .results import EvaluationResults
from .evaluator import MRAGBenchEvaluator, PerspectiveChangeType, ScenarioMetrics, EvaluationSession

__all__ = [
    "EvaluationResults",
    "MRAGBenchEvaluator",
    "PerspectiveChangeType",
    "ScenarioMetrics",
    "EvaluationSession"
]