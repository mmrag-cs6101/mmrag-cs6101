"""
Memory-Efficient Data Loading

Streaming data loader with memory management and batch optimization for MRAG-Bench.
"""

import logging
import gc
from typing import Iterator, List, Dict, Any, Optional, Tuple, Generator
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from dataclasses import dataclass
import json
import time

from .interface import Sample, BatchData
from .mrag_dataset import MRAGDataset
from .preprocessing import ImagePreprocessor, PreprocessingConfig, BatchDataProcessor
try:
    from ..utils.memory_manager import MemoryManager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming data loader."""
    chunk_size: int = 100
    prefetch_size: int = 2
    enable_caching: bool = True
    cache_size_mb: int = 500
    memory_threshold_gb: float = 14.0
    auto_batch_sizing: bool = True


class StreamingMRAGDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for MRAG-Bench.

    Provides:
    - Lazy loading of images
    - Memory-aware batching
    - Scenario-specific streaming
    - Automatic garbage collection
    """

    def __init__(
        self,
        dataset: MRAGDataset,
        scenario_type: str,
        config: StreamingConfig,
        preprocessor: ImagePreprocessor,
        memory_manager: MemoryManager
    ):
        """
        Initialize streaming dataset.

        Args:
            dataset: Base MRAG dataset
            scenario_type: Perspective change scenario to stream
            config: Streaming configuration
            preprocessor: Image preprocessor
            memory_manager: Memory manager instance
        """
        self.dataset = dataset
        self.scenario_type = scenario_type
        self.config = config
        self.preprocessor = preprocessor
        self.memory_manager = memory_manager

        # Cache for loaded samples
        self._sample_cache = {}
        self._cache_size_bytes = 0

        logger.info(f"StreamingMRAGDataset initialized for scenario: {scenario_type}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples with memory management."""
        sample_count = 0

        for sample in self.dataset.load_scenario(self.scenario_type):
            # Check memory before processing
            if self.memory_manager.check_memory_pressure():
                logger.warning("Memory pressure detected, clearing cache")
                self._clear_cache()
                gc.collect()

            # Load and process sample
            processed_sample = self._process_sample(sample)
            if processed_sample:
                yield processed_sample
                sample_count += 1

                # Periodic memory cleanup
                if sample_count % self.config.chunk_size == 0:
                    self._cleanup_memory()

    def _process_sample(self, sample: Sample) -> Optional[Dict[str, Any]]:
        """Process a single sample with caching."""
        try:
            # Check cache first
            cache_key = sample.question_id
            if self.config.enable_caching and cache_key in self._sample_cache:
                return self._sample_cache[cache_key]

            # Load image if needed
            if sample.image is None and sample.image_path:
                sample.image = self._load_image_safely(sample.image_path)

            # Preprocess image
            if sample.image:
                image_tensor = self.preprocessor.preprocess_image(
                    sample.image,
                    scenario_type=sample.perspective_type
                )
            else:
                # Create placeholder
                image_tensor = torch.zeros(3, *self.preprocessor.config.image_size)

            processed_sample = {
                'image': image_tensor,
                'question': sample.question,
                'question_id': sample.question_id,
                'ground_truth': sample.ground_truth,
                'perspective_type': sample.perspective_type,
                'metadata': sample.metadata
            }

            # Cache if enabled and within memory limits
            if self.config.enable_caching and self._should_cache():
                self._sample_cache[cache_key] = processed_sample
                self._update_cache_size(processed_sample)

            return processed_sample

        except Exception as e:
            logger.error(f"Error processing sample {sample.question_id}: {e}")
            return None

    def _load_image_safely(self, image_path: str) -> Optional[Any]:
        """Safely load image with error handling."""
        try:
            from PIL import Image
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    def _should_cache(self) -> bool:
        """Check if we should cache more items."""
        cache_size_mb = self._cache_size_bytes / (1024 * 1024)
        return cache_size_mb < self.config.cache_size_mb

    def _update_cache_size(self, sample: Dict[str, Any]) -> None:
        """Update cache size estimate."""
        # Rough estimate of sample size
        image_size = sample['image'].numel() * sample['image'].element_size()
        text_size = len(sample['question']) * 4  # Rough estimate
        self._cache_size_bytes += image_size + text_size

    def _clear_cache(self) -> None:
        """Clear sample cache."""
        self._sample_cache.clear()
        self._cache_size_bytes = 0

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        self.memory_manager.clear_gpu_memory()
        gc.collect()


class MemoryAwareDataLoader:
    """
    Memory-aware data loader with dynamic batch sizing and streaming support.

    Features:
    - Dynamic batch size adjustment based on available memory
    - Memory monitoring and automatic cleanup
    - Efficient streaming for large datasets
    - Scenario-specific data loading
    """

    def __init__(
        self,
        dataset: MRAGDataset,
        streaming_config: StreamingConfig,
        preprocessing_config: PreprocessingConfig,
        memory_manager: Optional[MemoryManager] = None
    ):
        """
        Initialize memory-aware data loader.

        Args:
            dataset: MRAG dataset instance
            streaming_config: Streaming configuration
            preprocessing_config: Preprocessing configuration
            memory_manager: Optional memory manager
        """
        self.dataset = dataset
        self.streaming_config = streaming_config
        self.preprocessing_config = preprocessing_config
        self.memory_manager = memory_manager or MemoryManager()

        # Initialize components
        self.preprocessor = ImagePreprocessor(preprocessing_config, self.memory_manager)
        self.batch_processor = BatchDataProcessor(self.preprocessor, self.memory_manager)

        # Performance tracking
        self.performance_stats = {
            "samples_processed": 0,
            "batches_created": 0,
            "memory_warnings": 0,
            "processing_time": 0.0
        }

        logger.info("MemoryAwareDataLoader initialized")

    def create_streaming_loader(
        self,
        scenario_type: str,
        batch_size: int = 8,
        shuffle: bool = False
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Create streaming data loader for a specific scenario.

        Args:
            scenario_type: Perspective change scenario type
            batch_size: Base batch size (will be adjusted based on memory)
            shuffle: Whether to shuffle data (limited support in streaming)

        Yields:
            Batches of processed data
        """
        start_time = time.time()

        # Create streaming dataset
        streaming_dataset = StreamingMRAGDataset(
            dataset=self.dataset,
            scenario_type=scenario_type,
            config=self.streaming_config,
            preprocessor=self.preprocessor,
            memory_manager=self.memory_manager
        )

        # Adjust batch size based on memory
        if self.streaming_config.auto_batch_sizing:
            batch_size = self.batch_processor.get_optimal_batch_size(
                batch_size, self.preprocessing_config.image_size
            )
            logger.info(f"Adjusted batch size to {batch_size} based on available memory")

        current_batch = []

        with self.memory_manager.memory_guard(f"streaming_loader_{scenario_type}"):
            for sample in streaming_dataset:
                current_batch.append(sample)

                if len(current_batch) >= batch_size:
                    # Process batch
                    batch_data = self._create_batch_from_samples(current_batch)
                    if batch_data:
                        yield batch_data
                        self.performance_stats["batches_created"] += 1

                    # Reset batch
                    current_batch = []

                    # Memory check
                    if self.memory_manager.check_memory_pressure():
                        self.performance_stats["memory_warnings"] += 1
                        self.memory_manager.emergency_cleanup()

                self.performance_stats["samples_processed"] += 1

            # Process remaining samples in last batch
            if current_batch:
                batch_data = self._create_batch_from_samples(current_batch)
                if batch_data:
                    yield batch_data
                    self.performance_stats["batches_created"] += 1

        self.performance_stats["processing_time"] += time.time() - start_time

    def create_standard_loader(
        self,
        scenario_type: str,
        batch_size: int = 8,
        shuffle: bool = False,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create standard PyTorch DataLoader.

        Args:
            scenario_type: Perspective change scenario type
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes

        Returns:
            PyTorch DataLoader
        """
        # Adjust batch size for memory
        if self.streaming_config.auto_batch_sizing:
            batch_size = self.batch_processor.get_optimal_batch_size(
                batch_size, self.preprocessing_config.image_size
            )

        return self.dataset.create_dataloader(
            scenario_type=scenario_type,
            shuffle=shuffle
        )

    def _create_batch_from_samples(self, samples: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        """Create a batch tensor from list of samples."""
        try:
            if not samples:
                return None

            batch_data = {
                'images': torch.stack([s['image'] for s in samples]),
                'questions': [s['question'] for s in samples],
                'question_ids': [s['question_id'] for s in samples],
                'ground_truths': [s['ground_truth'] for s in samples],
                'perspective_types': [s['perspective_type'] for s in samples],
                'metadata': [s['metadata'] for s in samples]
            }

            return batch_data

        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            return None

    def get_scenario_loader(
        self,
        scenario_type: str,
        streaming: bool = True,
        **kwargs
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Get data loader for specific scenario.

        Args:
            scenario_type: Perspective change scenario type
            streaming: Whether to use streaming loader
            **kwargs: Additional arguments for loader

        Returns:
            Data loader (streaming or standard)
        """
        if streaming:
            return self.create_streaming_loader(scenario_type, **kwargs)
        else:
            return self.create_standard_loader(scenario_type, **kwargs)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        stats["memory_stats"] = self.memory_manager.monitor.get_current_stats()
        stats["memory_trend"] = self.memory_manager.monitor.get_memory_usage_trend()
        return stats

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            "samples_processed": 0,
            "batches_created": 0,
            "memory_warnings": 0,
            "processing_time": 0.0
        }

    def validate_memory_efficiency(self, scenario_type: str, test_batches: int = 10) -> Dict[str, Any]:
        """
        Validate memory efficiency of data loading.

        Args:
            scenario_type: Scenario to test
            test_batches: Number of batches to test

        Returns:
            Validation results
        """
        start_memory = self.memory_manager.monitor.get_current_stats()
        start_time = time.time()

        batch_count = 0
        max_memory = start_memory.gpu_allocated_gb

        try:
            loader = self.create_streaming_loader(scenario_type, batch_size=4)

            for batch in loader:
                current_memory = self.memory_manager.monitor.get_current_stats()
                max_memory = max(max_memory, current_memory.gpu_allocated_gb)

                batch_count += 1
                if batch_count >= test_batches:
                    break

            end_time = time.time()
            end_memory = self.memory_manager.monitor.get_current_stats()

            return {
                "status": "success",
                "batches_processed": batch_count,
                "processing_time": end_time - start_time,
                "memory_start_gb": start_memory.gpu_allocated_gb,
                "memory_end_gb": end_memory.gpu_allocated_gb,
                "memory_peak_gb": max_memory,
                "memory_increase_gb": end_memory.gpu_allocated_gb - start_memory.gpu_allocated_gb,
                "avg_time_per_batch": (end_time - start_time) / max(batch_count, 1),
                "memory_stable": abs(end_memory.gpu_allocated_gb - start_memory.gpu_allocated_gb) < 0.1
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "batches_processed": batch_count
            }