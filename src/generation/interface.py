"""
Generation Pipeline Interface

Abstract base class for LLaVA-based multimodal answer generation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class MultimodalContext:
    """Multimodal context for generation."""
    question: str
    images: List[Image.Image]
    image_paths: List[str] = None
    context_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.image_paths is None:
            self.image_paths = []
        if self.context_metadata is None:
            self.context_metadata = {}


@dataclass
class GenerationResult:
    """Generation result with metadata."""
    answer: str
    confidence_score: float = 0.0
    generation_time: float = 0.0
    memory_usage: Dict[str, float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.memory_usage is None:
            self.memory_usage = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GenerationConfig:
    """Configuration for generation pipeline."""
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    quantization: str = "4bit"  # "4bit", "8bit", or "none"
    max_memory_gb: float = 5.0
    device: str = "cuda"
    torch_dtype: str = "float16"


class GenerationPipeline(ABC):
    """Abstract interface for multimodal generation pipeline."""

    def __init__(self, config: GenerationConfig):
        """
        Initialize generation pipeline.

        Args:
            config: Generation configuration parameters
        """
        self.config = config
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        """Load and initialize the vision-language model with quantization."""
        pass

    @abstractmethod
    def generate_answer(self, context: MultimodalContext) -> GenerationResult:
        """
        Generate answer for multimodal context.

        Args:
            context: MultimodalContext with question and retrieved images

        Returns:
            GenerationResult with answer and metadata
        """
        pass

    @abstractmethod
    def construct_prompt(self, context: MultimodalContext) -> str:
        """
        Construct prompt from multimodal context.

        Args:
            context: MultimodalContext with question and images

        Returns:
            Formatted prompt string for the model
        """
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        """Clear GPU memory and release model resources."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model from memory for dynamic loading."""
        pass

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage in GB
        """
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved": torch.cuda.memory_reserved() / 1024**3,
                "gpu_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return {"gpu_allocated": 0.0, "gpu_reserved": 0.0, "gpu_total": 0.0}

    def validate_memory_constraints(self) -> bool:
        """
        Check if current memory usage is within configured limits.

        Returns:
            True if within limits, False otherwise
        """
        memory_stats = self.get_memory_usage()
        return memory_stats["gpu_allocated"] <= self.config.max_memory_gb