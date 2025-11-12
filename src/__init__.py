"""
MRAG-Bench Reproduction System

A modular multimodal retrieval-augmented generation system for reproducing
MRAG-Bench baseline results on perspective change scenarios in medical imaging.

Architecture:
- dataset: MRAG-Bench data loading and preprocessing
- retrieval: CLIP-based image retrieval with FAISS indexing
- generation: LLaVA-based answer generation with quantization
- evaluation: MRAG-Bench evaluation framework and metrics

Requirements:
- RTX 5070Ti GPU (16GB VRAM)
- Target accuracy: 53-59% on perspective change scenarios
- Memory constraint: d15GB VRAM operation
"""

__version__ = "1.0.0"
__author__ = "AI Engineer"

# Core components
from .dataset import DatasetInterface
from .retrieval import RetrievalPipeline
from .generation import GenerationPipeline
from .evaluation import EvaluationResults

__all__ = [
    "DatasetInterface",
    "RetrievalPipeline",
    "GenerationPipeline",
    "EvaluationResults"
]