"""
Generation Module

Provides LLaVA-based multimodal answer generation with 4-bit quantization.
Implements vision-language model integration and memory-optimized inference.
"""

from .interface import GenerationPipeline
from .vlm_model import VisionLanguageModel

__all__ = ["GenerationPipeline", "VisionLanguageModel"]