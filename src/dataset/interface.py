"""
Dataset Interface

Abstract base class for MRAG-Bench dataset access with perspective change filtering.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class Sample:
    """Single MRAG-Bench sample."""
    question_id: str
    question: str
    image_path: str
    image: Optional[Image.Image] = None
    ground_truth: str = ""
    perspective_type: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchData:
    """Batch of preprocessed samples."""
    samples: List[Sample]
    images: List[Image.Image]
    questions: List[str]
    batch_size: int

    def __post_init__(self):
        self.batch_size = len(self.samples)


class DatasetInterface(ABC):
    """Abstract interface for MRAG-Bench dataset access."""

    def __init__(self, data_path: str, batch_size: int = 8):
        """
        Initialize dataset interface.

        Args:
            data_path: Path to MRAG-Bench dataset
            batch_size: Batch size for streaming operations
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.perspective_types = ['angle', 'partial', 'scope', 'occlusion']

    @abstractmethod
    def load_scenario(self, scenario_type: str) -> Iterator[Sample]:
        """
        Load samples for specific perspective change scenario.

        Args:
            scenario_type: One of ['angle', 'partial', 'scope', 'occlusion']

        Yields:
            Sample objects for the specified scenario
        """
        pass

    @abstractmethod
    def get_retrieval_corpus(self) -> List[str]:
        """
        Get image paths for retrieval corpus.

        Returns:
            List of image file paths for building retrieval index
        """
        pass

    @abstractmethod
    def preprocess_batch(self, samples: List[Sample]) -> BatchData:
        """
        Preprocess a batch of samples for model input.

        Args:
            samples: List of Sample objects

        Returns:
            BatchData with preprocessed images and questions
        """
        pass

    @abstractmethod
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate dataset integrity and return statistics.

        Returns:
            Dictionary with validation results and dataset statistics
        """
        pass

    def get_scenario_stats(self) -> Dict[str, int]:
        """
        Get sample count for each perspective change scenario.

        Returns:
            Dictionary mapping scenario types to sample counts
        """
        stats = {}
        for scenario in self.perspective_types:
            count = sum(1 for _ in self.load_scenario(scenario))
            stats[scenario] = count
        return stats