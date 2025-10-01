"""
Embedding Processor for MRAG-Bench Image Corpus

Handles batch processing of large image collections for embedding generation.
Optimized for memory efficiency and processing the complete MRAG-Bench dataset.
"""

import os
import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Iterator
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .clip_retriever import CLIPRetriever
from .interface import RetrievalConfig
from ..dataset.interface import DatasetInterface
from ..utils.memory_manager import MemoryManager
from ..utils.error_handling import handle_errors, MRAGError


logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Processes large image collections to generate embeddings with memory management.

    Features:
    - Batch processing with adaptive batch sizing
    - Memory-aware processing to stay within VRAM limits
    - Progress tracking and resumption capabilities
    - Embedding caching and validation
    - Performance monitoring and optimization
    """

    def __init__(
        self,
        retriever: CLIPRetriever,
        memory_manager: Optional[MemoryManager] = None,
        cache_dir: str = "data/embeddings"
    ):
        """
        Initialize embedding processor.

        Args:
            retriever: CLIPRetriever instance
            memory_manager: Memory manager instance
            cache_dir: Directory for caching embeddings
        """
        self.retriever = retriever
        self.memory_manager = memory_manager or MemoryManager()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Processing state
        self.processed_paths = set()
        self.failed_paths = set()
        self.embeddings_cache = {}

        # Performance tracking
        self.stats = {
            "total_images_processed": 0,
            "total_processing_time": 0.0,
            "avg_batch_time": 0.0,
            "memory_optimizations": 0,
            "failed_images": 0
        }

        logger.info(f"EmbeddingProcessor initialized with cache dir: {cache_dir}")

    @handle_errors
    def process_image_corpus(
        self,
        image_paths: List[str],
        batch_size: int = 16,
        max_images: Optional[int] = None,
        resume_from_cache: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process complete image corpus to generate embeddings.

        Args:
            image_paths: List of image file paths
            batch_size: Initial batch size (will be adapted based on memory)
            max_images: Maximum number of images to process (for testing)
            resume_from_cache: Whether to resume from existing cache

        Returns:
            Tuple of (embeddings array, valid image paths)
        """
        if max_images is not None:
            image_paths = image_paths[:max_images]

        logger.info(f"Processing {len(image_paths)} images for embedding generation...")

        # Load existing cache if resuming
        if resume_from_cache:
            self._load_cache()

        # Filter already processed images
        remaining_paths = [
            path for path in image_paths
            if path not in self.processed_paths and path not in self.failed_paths
        ]

        logger.info(f"Found {len(remaining_paths)} new images to process")

        if not remaining_paths and self.processed_paths:
            logger.info("All images already processed, loading from cache")
            return self._load_embeddings_from_cache(image_paths)

        # Warm up the model
        self.retriever.warmup()

        # Process images in batches
        all_embeddings = []
        valid_paths = []

        start_time = time.time()
        current_batch_size = batch_size

        try:
            with tqdm(total=len(remaining_paths), desc="Processing images") as pbar:
                for batch_start in range(0, len(remaining_paths), current_batch_size):
                    batch_end = min(batch_start + current_batch_size, len(remaining_paths))
                    batch_paths = remaining_paths[batch_start:batch_end]

                    # Process batch
                    batch_embeddings, batch_valid_paths = self._process_batch(
                        batch_paths, pbar
                    )

                    if batch_embeddings.size > 0:
                        all_embeddings.append(batch_embeddings)
                        valid_paths.extend(batch_valid_paths)

                    # Adaptive batch sizing based on memory pressure
                    if self.memory_manager.monitor.check_memory_pressure():
                        current_batch_size = max(1, current_batch_size // 2)
                        self.stats["memory_optimizations"] += 1
                        logger.warning(f"Reducing batch size to {current_batch_size} due to memory pressure")

                    # Save progress periodically
                    if len(all_embeddings) % 10 == 0:
                        self._save_cache()

        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            # Save progress before re-raising
            self._save_cache()
            raise

        # Combine all embeddings
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
        else:
            final_embeddings = np.empty((0, self.retriever.config.embedding_dim))

        # Final cache save
        self._save_cache()

        # Update stats
        total_time = time.time() - start_time
        self.stats["total_processing_time"] = total_time
        self.stats["total_images_processed"] = len(valid_paths)

        logger.info(
            f"Embedding processing completed. "
            f"Processed {len(valid_paths)} images in {total_time:.2f}s. "
            f"Final embeddings shape: {final_embeddings.shape}"
        )

        return final_embeddings, valid_paths

    @handle_errors
    def _process_batch(self, batch_paths: List[str], pbar: tqdm) -> Tuple[np.ndarray, List[str]]:
        """Process a single batch of images."""
        batch_start_time = time.time()

        with self.memory_manager.memory_guard(f"Batch processing ({len(batch_paths)} images)"):
            # Load images
            loaded_images = []
            valid_paths = []

            for image_path in batch_paths:
                try:
                    if os.path.exists(image_path):
                        image = Image.open(image_path).convert('RGB')
                        loaded_images.append(image)
                        valid_paths.append(image_path)
                    else:
                        logger.warning(f"Image not found: {image_path}")
                        self.failed_paths.add(image_path)
                        self.stats["failed_images"] += 1

                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
                    self.failed_paths.add(image_path)
                    self.stats["failed_images"] += 1

            if not loaded_images:
                pbar.update(len(batch_paths))
                return np.empty((0, self.retriever.config.embedding_dim)), []

            # Generate embeddings
            try:
                embeddings = self.retriever.encode_images(loaded_images)

                # Cache embeddings
                for i, path in enumerate(valid_paths):
                    self.embeddings_cache[path] = embeddings[i]
                    self.processed_paths.add(path)

                # Update progress
                pbar.update(len(batch_paths))

                # Update timing stats
                batch_time = time.time() - batch_start_time
                self.stats["avg_batch_time"] = (
                    self.stats["avg_batch_time"] * 0.9 + batch_time * 0.1
                )

                return embeddings, valid_paths

            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Mark all images in batch as failed
                for path in batch_paths:
                    self.failed_paths.add(path)
                    self.stats["failed_images"] += 1

                pbar.update(len(batch_paths))
                return np.empty((0, self.retriever.config.embedding_dim)), []

    def _load_cache(self) -> None:
        """Load existing embedding cache."""
        cache_file = self.cache_dir / "embeddings_cache.json"
        processed_file = self.cache_dir / "processed_paths.json"
        failed_file = self.cache_dir / "failed_paths.json"

        try:
            # Load processed paths
            if processed_file.exists():
                with open(processed_file, 'r') as f:
                    data = json.load(f)
                    self.processed_paths = set(data.get("processed_paths", []))
                    logger.info(f"Loaded {len(self.processed_paths)} processed paths from cache")

            # Load failed paths
            if failed_file.exists():
                with open(failed_file, 'r') as f:
                    data = json.load(f)
                    self.failed_paths = set(data.get("failed_paths", []))
                    logger.info(f"Loaded {len(self.failed_paths)} failed paths from cache")

            # Load embedding vectors
            embeddings_file = self.cache_dir / "embeddings.npy"
            paths_file = self.cache_dir / "embedding_paths.json"

            if embeddings_file.exists() and paths_file.exists():
                embeddings = np.load(embeddings_file)
                with open(paths_file, 'r') as f:
                    paths = json.load(f)

                # Rebuild embeddings cache
                if len(paths) == embeddings.shape[0]:
                    for i, path in enumerate(paths):
                        self.embeddings_cache[path] = embeddings[i]
                    logger.info(f"Loaded {len(paths)} embeddings from cache")

        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            self.processed_paths = set()
            self.failed_paths = set()
            self.embeddings_cache = {}

    def _save_cache(self) -> None:
        """Save current processing state to cache."""
        try:
            # Save processed and failed paths
            processed_file = self.cache_dir / "processed_paths.json"
            with open(processed_file, 'w') as f:
                json.dump({"processed_paths": list(self.processed_paths)}, f, indent=2)

            failed_file = self.cache_dir / "failed_paths.json"
            with open(failed_file, 'w') as f:
                json.dump({"failed_paths": list(self.failed_paths)}, f, indent=2)

            # Save embeddings if we have any
            if self.embeddings_cache:
                paths = list(self.embeddings_cache.keys())
                embeddings = np.array([self.embeddings_cache[path] for path in paths])

                embeddings_file = self.cache_dir / "embeddings.npy"
                np.save(embeddings_file, embeddings)

                paths_file = self.cache_dir / "embedding_paths.json"
                with open(paths_file, 'w') as f:
                    json.dump(paths, f, indent=2)

                logger.debug(f"Saved {len(paths)} embeddings to cache")

        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _load_embeddings_from_cache(self, requested_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Load embeddings for requested paths from cache."""
        valid_embeddings = []
        valid_paths = []

        for path in requested_paths:
            if path in self.embeddings_cache:
                valid_embeddings.append(self.embeddings_cache[path])
                valid_paths.append(path)
            elif path not in self.failed_paths:
                logger.warning(f"Path not in cache and not failed: {path}")

        if valid_embeddings:
            return np.array(valid_embeddings), valid_paths
        else:
            return np.empty((0, self.retriever.config.embedding_dim)), []

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_images": len(self.processed_paths),
            "failed_images": len(self.failed_paths),
            "cached_embeddings": len(self.embeddings_cache),
            "performance_stats": self.stats,
            "memory_usage": self.memory_manager.monitor.get_current_stats()
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        logger.info("Clearing embedding cache...")

        self.processed_paths.clear()
        self.failed_paths.clear()
        self.embeddings_cache.clear()

        # Remove cache files
        cache_files = [
            "processed_paths.json",
            "failed_paths.json",
            "embeddings.npy",
            "embedding_paths.json"
        ]

        for filename in cache_files:
            cache_file = self.cache_dir / filename
            if cache_file.exists():
                cache_file.unlink()

        logger.info("Embedding cache cleared")

    def validate_embeddings(self, embeddings: np.ndarray, image_paths: List[str]) -> Dict[str, Any]:
        """
        Validate generated embeddings.

        Args:
            embeddings: Generated embeddings array
            image_paths: Corresponding image paths

        Returns:
            Validation results
        """
        validation_results = {
            "status": "success",
            "total_embeddings": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1] if embeddings.ndim > 1 else 0,
            "path_count": len(image_paths),
            "errors": []
        }

        try:
            # Check shape consistency
            if embeddings.shape[0] != len(image_paths):
                validation_results["errors"].append(
                    f"Embedding count ({embeddings.shape[0]}) != path count ({len(image_paths)})"
                )

            # Check embedding dimension
            expected_dim = self.retriever.config.embedding_dim
            if embeddings.shape[1] != expected_dim:
                validation_results["errors"].append(
                    f"Embedding dimension ({embeddings.shape[1]}) != expected ({expected_dim})"
                )

            # Check for invalid values
            if np.any(np.isnan(embeddings)):
                nan_count = np.sum(np.isnan(embeddings))
                validation_results["errors"].append(f"Found {nan_count} NaN values in embeddings")

            if np.any(np.isinf(embeddings)):
                inf_count = np.sum(np.isinf(embeddings))
                validation_results["errors"].append(f"Found {inf_count} infinite values in embeddings")

            # Check normalization
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                validation_results["errors"].append("Embeddings are not properly normalized")

            # Set status based on errors
            if validation_results["errors"]:
                validation_results["status"] = "warning" if len(validation_results["errors"]) < 3 else "error"

            logger.info(f"Embedding validation: {validation_results['status']}")

        except Exception as e:
            validation_results["status"] = "error"
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            logger.error(f"Embedding validation failed: {e}")

        return validation_results