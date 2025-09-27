"""
Generation Module

Provides LLaVA-based multimodal answer generation with 4-bit quantization.
Implements vision-language model integration and memory-optimized inference.
"""

from .interface import GenerationPipeline

__all__ = ["GenerationPipeline"]