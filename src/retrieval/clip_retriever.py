"""
CLIP-based Image Retrieval Pipeline

Implements CLIP ViT-B/32 based image retrieval with FAISS indexing for MRAG-Bench system.
Optimized for 16GB VRAM constraints with aggressive memory management.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import faiss

from .interface import RetrievalPipeline, RetrievalResult, RetrievalConfig
from ..utils.memory_manager import MemoryManager
from ..utils.error_handling import handle_errors, MRAGError


logger = logging.getLogger(__name__)


class CLIPRetriever(RetrievalPipeline):
    """
    CLIP ViT-B/32 based image retrieval pipeline with FAISS indexing.

    Features:
    - Memory-optimized CLIP model loading
    - Batch embedding generation with VRAM management
    - FAISS vector storage and similarity search
    - Configurable top-k retrieval with similarity scoring
    - Persistent index caching for fast loading
    """

    def __init__(self, config: RetrievalConfig):
        """
        Initialize CLIP retrieval pipeline.

        Args:
            config: Retrieval configuration parameters
        """
        super().__init__(config)

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.index = None
        self.image_paths = []
        self.embeddings = None

        # Memory management
        self.memory_manager = MemoryManager(
            memory_limit_gb=config.max_memory_gb + 10,  # Total system limit
            buffer_gb=1.0
        )

        # Performance tracking
        self.stats = {
            "total_embeddings_generated": 0,
            "total_queries_processed": 0,
            "avg_encoding_time": 0.0,
            "avg_retrieval_time": 0.0
        }

        logger.info(f"CLIPRetriever initialized for device: {self.device}")

    @handle_errors
    def _load_model(self) -> None:
        """Load CLIP model with memory optimization."""
        if self.model is not None:
            return

        with self.memory_manager.memory_guard("CLIP model loading"):
            logger.info(f"Loading CLIP model: {self.config.model_name}")

            # Load processor
            self.processor = CLIPProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=os.path.expanduser("~/.cache/huggingface/transformers")
            )

            # Load model with memory optimization
            self.model = CLIPModel.from_pretrained(
                self.config.model_name,
                cache_dir=os.path.expanduser("~/.cache/huggingface/transformers"),
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map=self.device if self.device.type == "cuda" else None
            )

            # Move to device if not already there
            if not hasattr(self.model, 'device') or self.model.device != self.device:
                self.model = self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            # Log memory usage
            memory_stats = self.memory_manager.monitor.log_memory_stats("After CLIP loading")
            logger.info(
                f"CLIP model loaded successfully. "
                f"GPU memory: {memory_stats.gpu_allocated_gb:.2f}GB"
            )

    @handle_errors
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate embeddings for a batch of images.

        Args:
            images: List of PIL images

        Returns:
            Numpy array of embeddings (batch_size, embedding_dim)
        """
        if not images:
            return np.empty((0, self.config.embedding_dim))

        self._load_model()

        start_time = time.time()

        with self.memory_manager.memory_guard("Image encoding"):
            # Preprocess images
            try:
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding=True
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                    # Normalize embeddings
                    image_features = F.normalize(image_features, p=2, dim=1)

                # Convert to numpy
                embeddings = image_features.cpu().numpy().astype(np.float32)

                # Update stats
                encoding_time = time.time() - start_time
                self.stats["total_embeddings_generated"] += len(images)
                self.stats["avg_encoding_time"] = (
                    self.stats["avg_encoding_time"] * 0.9 + encoding_time * 0.1
                )

                logger.debug(
                    f"Encoded {len(images)} images in {encoding_time:.2f}s. "
                    f"Embeddings shape: {embeddings.shape}"
                )

                return embeddings

            except Exception as e:
                logger.error(f"Error encoding images: {e}")
                # Return zero embeddings as fallback
                return np.zeros((len(images), self.config.embedding_dim), dtype=np.float32)

    @handle_errors
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of text queries.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (batch_size, embedding_dim)
        """
        if not texts:
            return np.empty((0, self.config.embedding_dim))

        self._load_model()

        with self.memory_manager.memory_guard("Text encoding"):
            try:
                # Preprocess text
                inputs = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77  # CLIP's max text length
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)

                    # Normalize embeddings
                    text_features = F.normalize(text_features, p=2, dim=1)

                # Convert to numpy
                embeddings = text_features.cpu().numpy().astype(np.float32)

                logger.debug(f"Encoded {len(texts)} texts. Embeddings shape: {embeddings.shape}")

                return embeddings

            except Exception as e:
                logger.error(f"Error encoding texts: {e}")
                # Return zero embeddings as fallback
                return np.zeros((len(texts), self.config.embedding_dim), dtype=np.float32)

    @handle_errors
    def build_index(self, embeddings: np.ndarray, image_paths: List[str]) -> None:
        """
        Build FAISS index from image embeddings.

        Args:
            embeddings: Image embeddings array (num_images, embedding_dim)
            image_paths: Corresponding image file paths
        """
        if embeddings.shape[0] == 0:
            raise MRAGError("Cannot build index from empty embeddings")

        if len(image_paths) != embeddings.shape[0]:
            raise MRAGError(
                f"Mismatch between embeddings ({embeddings.shape[0]}) "
                f"and image paths ({len(image_paths)})"
            )

        logger.info(f"Building FAISS index for {embeddings.shape[0]} embeddings...")

        with self.memory_manager.memory_guard("FAISS index building"):
            # Ensure embeddings are float32 and normalized
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            # Normalize embeddings if not already normalized
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if not np.allclose(norms, 1.0, atol=1e-3):
                embeddings = embeddings / norms

            # Choose index type based on corpus size
            num_embeddings = embeddings.shape[0]

            if num_embeddings < 1000:
                # Use exact search for small corpus
                self.index = faiss.IndexFlatIP(self.config.embedding_dim)
                logger.info("Using exact FAISS index (IndexFlatIP)")
            else:
                # Use IVF index for larger corpus
                nlist = min(int(np.sqrt(num_embeddings)), 1024)  # Number of clusters
                quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.config.embedding_dim, nlist)

                # Train the index
                logger.info(f"Training IVF index with {nlist} clusters...")
                self.index.train(embeddings)
                logger.info("Using IVF FAISS index (IndexIVFFlat)")

            # Add embeddings to index
            self.index.add(embeddings)

            # Configure IVF index search parameters
            if isinstance(self.index, faiss.IndexIVFFlat):
                # Set nprobe to search more clusters for better recall
                # Use 10% of clusters or at least 10
                self.index.nprobe = max(10, nlist // 10)
                logger.info(f"Set IVF nprobe to {self.index.nprobe} (searching {self.index.nprobe}/{nlist} clusters)")

            # Store image paths and embeddings
            self.image_paths = image_paths.copy()
            self.embeddings = embeddings.copy()

            logger.info(
                f"FAISS index built successfully. "
                f"Total vectors: {self.index.ntotal}, "
                f"Index type: {type(self.index).__name__}"
            )

    @handle_errors
    def retrieve_similar(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve most similar images for a text query.

        Args:
            query: Text query string
            k: Number of results to return (defaults to config.top_k)

        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        if self.index is None:
            raise MRAGError("Index not built. Call build_index() first.")

        if k is None:
            k = self.config.top_k

        start_time = time.time()

        with self.memory_manager.memory_guard("Similarity retrieval"):
            # Encode query text
            query_embedding = self.encode_text([query])

            if query_embedding.shape[0] == 0:
                logger.warning("Failed to encode query, returning empty results")
                return []

            # Search index
            scores, indices = self.index.search(query_embedding, k)

            # Create results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])

                # Skip invalid indices
                if idx < 0 or idx >= len(self.image_paths):
                    continue

                # Apply similarity threshold
                if score < self.config.similarity_threshold:
                    continue

                result = RetrievalResult(
                    image_path=self.image_paths[idx],
                    similarity_score=score,
                    embedding=self.embeddings[idx] if self.embeddings is not None else None,
                    metadata={
                        "index": idx,
                        "query": query
                    }
                )
                results.append(result)

            # Update stats
            retrieval_time = time.time() - start_time
            self.stats["total_queries_processed"] += 1
            self.stats["avg_retrieval_time"] = (
                self.stats["avg_retrieval_time"] * 0.9 + retrieval_time * 0.1
            )

            logger.debug(
                f"Retrieved {len(results)} results for query in {retrieval_time:.3f}s. "
                f"Top score: {results[0].similarity_score:.3f}" if results else "No results found"
            )

            return results

    @handle_errors
    def retrieve_by_image(self, query_image: Image.Image, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve most similar images for an image query (image-to-image retrieval).

        Args:
            query_image: PIL Image to use as query
            k: Number of results to return (defaults to config.top_k)

        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        if self.index is None:
            raise MRAGError("Index not built. Call build_index() first.")

        if k is None:
            k = self.config.top_k

        start_time = time.time()

        with self.memory_manager.memory_guard("Image similarity retrieval"):
            # Encode query image
            query_embedding = self.encode_images([query_image])

            if query_embedding.shape[0] == 0:
                logger.warning("Failed to encode query image, returning empty results")
                return []

            # Search index
            scores, indices = self.index.search(query_embedding, k)

            # Create results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])

                # Skip invalid indices
                if idx < 0 or idx >= len(self.image_paths):
                    continue

                # Apply similarity threshold
                if score < self.config.similarity_threshold:
                    continue

                result = RetrievalResult(
                    image_path=self.image_paths[idx],
                    similarity_score=score,
                    embedding=self.embeddings[idx] if self.embeddings is not None else None,
                    metadata={
                        "index": idx,
                        "query_type": "image"
                    }
                )
                results.append(result)

            # Update stats
            retrieval_time = time.time() - start_time
            self.stats["total_queries_processed"] += 1
            self.stats["avg_retrieval_time"] = (
                self.stats["avg_retrieval_time"] * 0.9 + retrieval_time * 0.1
            )

            logger.debug(
                f"Retrieved {len(results)} results for image query in {retrieval_time:.3f}s. "
                f"Top score: {results[0].similarity_score:.3f}" if results else "No results found"
            )

            return results

    @handle_errors
    def save_index(self, index_path: str) -> None:
        """
        Save FAISS index to disk.

        Args:
            index_path: Path to save index file
        """
        if self.index is None:
            raise MRAGError("No index to save. Call build_index() first.")

        index_dir = Path(index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save metadata
        metadata = {
            "image_paths": self.image_paths,
            "config": {
                "model_name": self.config.model_name,
                "embedding_dim": self.config.embedding_dim,
                "top_k": self.config.top_k
            },
            "stats": self.stats
        }

        metadata_path = index_path.replace('.bin', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings separately
        if self.embeddings is not None:
            embeddings_path = index_path.replace('.bin', '_embeddings.npy')
            np.save(embeddings_path, self.embeddings)

        logger.info(f"Index saved to {index_path}")

    @handle_errors
    def load_index(self, index_path: str, image_paths: List[str]) -> None:
        """
        Load FAISS index from disk.

        Args:
            index_path: Path to index file
            image_paths: Corresponding image file paths
        """
        if not os.path.exists(index_path):
            raise MRAGError(f"Index file not found: {index_path}")

        logger.info(f"Loading FAISS index from {index_path}")

        with self.memory_manager.memory_guard("Index loading"):
            # Load FAISS index
            self.index = faiss.read_index(index_path)

            # Load metadata if available
            metadata_path = index_path.replace('.bin', '_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Verify compatibility
                saved_model = metadata.get("config", {}).get("model_name", "")
                if saved_model and saved_model != self.config.model_name:
                    logger.warning(
                        f"Model mismatch: saved={saved_model}, current={self.config.model_name}"
                    )

                # Load stats
                if "stats" in metadata:
                    self.stats.update(metadata["stats"])

                # Use saved image paths if provided paths don't match
                saved_paths = metadata.get("image_paths", [])
                if len(saved_paths) == self.index.ntotal:
                    self.image_paths = saved_paths
                else:
                    self.image_paths = image_paths
            else:
                self.image_paths = image_paths

            # Load embeddings if available
            embeddings_path = index_path.replace('.bin', '_embeddings.npy')
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")

            logger.info(
                f"Index loaded successfully. "
                f"Total vectors: {self.index.ntotal}, "
                f"Image paths: {len(self.image_paths)}"
            )

    def clear_memory(self) -> None:
        """Clear GPU memory and release resources."""
        logger.info("Clearing CLIP retriever memory...")

        # Clear model
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear FAISS index (keep on CPU)
        # Index is kept in memory for fast access

        # Clear GPU memory
        self.memory_manager.clear_gpu_memory(aggressive=True)

        logger.info("CLIP retriever memory cleared")

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding and retrieval statistics."""
        index_info = {}
        if self.index is not None:
            index_info = {
                "total_vectors": self.index.ntotal,
                "index_type": type(self.index).__name__,
                "is_trained": getattr(self.index, 'is_trained', True)
            }

        return {
            "model_loaded": self.model is not None,
            "index_built": self.index is not None,
            "image_paths_count": len(self.image_paths),
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
            "index_info": index_info,
            "performance_stats": self.stats,
            "memory_usage": self.get_memory_usage()
        }

    def warmup(self, num_images: int = 4, num_texts: int = 2) -> Dict[str, float]:
        """
        Warm up the model with dummy data to optimize performance.

        Args:
            num_images: Number of dummy images for warmup
            num_texts: Number of dummy texts for warmup

        Returns:
            Warmup timing statistics
        """
        logger.info("Warming up CLIP retriever...")

        warmup_stats = {}

        # Create dummy images
        dummy_images = [
            Image.new('RGB', (224, 224), color=(i * 50, i * 50, i * 50))
            for i in range(num_images)
        ]

        # Create dummy texts
        dummy_texts = [f"dummy query {i}" for i in range(num_texts)]

        # Warmup image encoding
        start_time = time.time()
        _ = self.encode_images(dummy_images)
        warmup_stats["image_encoding_time"] = time.time() - start_time

        # Warmup text encoding
        start_time = time.time()
        _ = self.encode_text(dummy_texts)
        warmup_stats["text_encoding_time"] = time.time() - start_time

        logger.info(f"CLIP retriever warmed up. Stats: {warmup_stats}")
        return warmup_stats