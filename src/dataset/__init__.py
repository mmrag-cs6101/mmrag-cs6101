"""
Dataset Package

MRAG-Bench dataset processing and loading functionality.
"""

from .interface import DatasetInterface, Sample, BatchData
from .mrag_dataset import MRAGDataset
from .preprocessing import ImagePreprocessor, PreprocessingConfig, BatchDataProcessor
from .data_loader import MemoryAwareDataLoader, StreamingConfig, StreamingMRAGDataset
from .validation import DatasetValidator, ValidationResult

__all__ = [
    # Interfaces
    'DatasetInterface',
    'Sample',
    'BatchData',

    # Core dataset
    'MRAGDataset',

    # Preprocessing
    'ImagePreprocessor',
    'PreprocessingConfig',
    'BatchDataProcessor',

    # Data loading
    'MemoryAwareDataLoader',
    'StreamingConfig',
    'StreamingMRAGDataset',

    # Validation
    'DatasetValidator',
    'ValidationResult',
]