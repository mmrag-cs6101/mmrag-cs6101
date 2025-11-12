"""
Retrieval Pipeline Interface

Abstract base class for CLIP-based image retrieval with FAISS indexing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    image_path: str
    similarity_score: float
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    model_name: str = "openai/clip-vit-base-patch32"
    model_name_sam: str = "sam2.1_b.pt"
    embedding_dim: int = 512
    top_k: int = 5
    similarity_threshold: float = 0.0
    batch_size: int = 16
    max_memory_gb: float = 1.0
    device: str = "cuda"


class RetrievalPipeline(ABC):
    """Abstract interface for image retrieval pipeline."""

    def __init__(self, config: RetrievalConfig):
        """
        Initialize retrieval pipeline.

        Args:
            config: Retrieval configuration parameters
        """
        self.config = config
        self.index = None
        self.image_paths = []

    @abstractmethod
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate embeddings for a batch of images.

        Args:
            images: List of PIL images

        Returns:
            Numpy array of embeddings (batch_size, embedding_dim)
        """
        pass

    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of text queries.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (batch_size, embedding_dim)
        """
        pass

    @abstractmethod
    def build_index(self, embeddings: np.ndarray, image_paths: List[str]) -> None:
        """
        Build FAISS index from image embeddings.

        Args:
            embeddings: Image embeddings array (num_images, embedding_dim)
            image_paths: Corresponding image file paths
        """
        pass

    @abstractmethod
    def retrieve_similar(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve most similar images for a text query.

        Args:
            query: Text query string
            k: Number of results to return (defaults to config.top_k)

        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        pass

    @abstractmethod
    def save_index(self, index_path: str) -> None:
        """
        Save FAISS index to disk.

        Args:
            index_path: Path to save index file
        """
        pass

    @abstractmethod
    def load_index(self, index_path: str, image_paths: List[str]) -> None:
        """
        Load FAISS index from disk.

        Args:
            index_path: Path to index file
            image_paths: Corresponding image file paths
        """
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        """Clear GPU memory and release resources."""
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