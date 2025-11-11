"""
Generation Module

Provides LLaVA-based multimodal answer generation with 4-bit quantization.
Implements vision-language model integration and memory-optimized inference.
"""

from .interface import (
    GenerationPipeline,
    GenerationConfig,
    GenerationResult,
    MultimodalContext
)
from .llava_pipeline import LLaVAGenerationPipeline
from .llava_enhanced_pipeline import EnhancedLLaVAPipeline
from .factory import GenerationPipelineFactory, create_llava_pipeline

__all__ = [
    "GenerationPipeline",
    "GenerationConfig",
    "GenerationResult",
    "MultimodalContext",
    "LLaVAGenerationPipeline",
    "EnhancedLLaVAPipeline",
    "GenerationPipelineFactory",
    "create_llava_pipeline"
]