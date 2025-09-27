"""
Evaluation Module

Provides MRAG-Bench evaluation framework for accuracy calculation and performance analysis.
Implements evaluation methodology matching the original paper.
"""

from .evaluator import MRAGBenchEvaluator
from .results import EvaluationResults

__all__ = ["MRAGBenchEvaluator", "EvaluationResults"]