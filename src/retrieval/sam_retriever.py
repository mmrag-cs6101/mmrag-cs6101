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
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from ultralytics import SAM
import faiss

from .interface import RetrievalPipeline, RetrievalResult, RetrievalConfig
from ..utils.memory_manager import MemoryManager
from ..utils.error_handling import handle_errors, MRAGError
import traceback


logger = logging.getLogger(__name__)


class SAMRetriever(RetrievalPipeline):
    """
    CLIP ViT-B/32 based image retrieval pipeline with FAISS indexing.

    Features:
    - 
    - Memory-optimized CLIP model loading
    - Batch embedding generation with VRAM management
    - FAISS vector storage and MaxSim search
    - Configurable top-k retrieval with MaxSim scoring
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
        self.model_sam = None
        self.processor = None
        self.index = None
        self.image_paths = []
        self.global_embeddings = None
        self.region_image_paths = None
        self.region_embeddings = None

        # to_tensor
        self.to_tensor = T.ToTensor()

        # Memory management
        self.memory_manager = MemoryManager(
            memory_limit_gb=config.max_memory_gb + 15,  # Total system limit
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
        """Load CLIP, SAM models with memory optimization."""
        if self.model is not None and self.model_sam is not None:
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

            # Load SAM model, test-time optimisations are in model_sam.predict arguments
            self.model_sam = SAM(self.config.model_name_sam)

            # Move to device if not already there
            if not hasattr(self.model, 'device') or self.model.device != self.device:
                self.model_sam = self.model_sam.to(self.device)

            # Set to evaluation mode
            self.model_sam.eval()

            # Log memory usage
            memory_stats = self.memory_manager.monitor.log_memory_stats("After SAM loading")
            logger.info(
                f"SAM model loaded successfully. "
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

        images_masked = []
        mapping = []
        with self.memory_manager.memory_guard("Image encoding"):
            # Preprocess images
            try:
                for i in range(len(images)):
                    img = images[i]
                    mask = self.model_sam(img, half=True, conf=0.5, max_det=6, iou=0.2, verbose=False) # SAM-2 does not support batched inference

                    # Format for transforms and CLIP
                    mask = mask[0].masks
                    if mask is not None:
                        mask = mask.data.float().to(self.device)
                        areas = []
                        for mi in range(mask.shape[0]):
                            arr = mask[mi] > 0
                            ys, xs = torch.where(arr)
                            area = (xs.max()-xs.min())*(ys.max()-ys.min()) if xs.numel() and ys.numel() else 0
                            areas.append((area, mi))
                        areas.sort(reverse=True, key=lambda x: x[0])
                        
                        # Keep top 14 largest masks
                        mask = mask[[areas[i][1] for i in range(min(14, len(areas)))]]


                    img = self.to_tensor(img).to(self.device) # C x H x W
                    
                    if mask is not None and mask.shape[0] > 0:
                        # Combine masks, find inverse
                        mask_combined = torch.max(mask, dim=0).values
                        mask = torch.cat((mask, torch.ones_like(img[0]).unsqueeze(0)-mask_combined), dim=0)
                    else:
                        mask = torch.ones_like(img[0]).unsqueeze(0) # 1 x H x W
                    mask = torch.cat((torch.ones_like(img[0]).unsqueeze(0), mask), dim=0) # I x H x W
                    I, C, H, W = mask.shape[0], img.shape[0], img.shape[1], img.shape[2]
                    mask_expanded = mask.unsqueeze(1).expand(I, C, H, W)  # I x C x H x W
                    img_expanded = img.unsqueeze(0).expand(I, C, H, W)
                    img_expanded = img_expanded * mask_expanded
                    images_masked += [img_expanded[i] for i in range(img_expanded.shape[0])] # flat list 
                    mapping.append(I)
                
                # Concatenate all masked images for batch processing
                images_masked = [T.functional.to_pil_image(i) for i in images_masked]

                inputs = self.processor(
                    images=images_masked,
                    return_tensors="pt",
                    padding=True
                )

                logger.info(f"Input batch size (masked images): {inputs['pixel_values'].shape[0]}")

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    logger.info(f"Raw image features shape: {image_features.shape}")

                    # Normalize embeddings
                    image_features = F.normalize(image_features, p=2, dim=1)

                # Convert to numpy, extract per-image embeddings
                embeddings_list = []
                embeddings = image_features.cpu().numpy().astype(np.float32)
                mapping = np.cumsum([0]+mapping)
                for i in range(len(images)): # no. images
                    emb = embeddings[mapping[i]:mapping[i+1]]
                    pad_size = max(0, 16-emb.shape[0]) # pad to 16 regions
                    emb = np.pad(emb, ((0,pad_size),(0,0)), mode='constant', constant_values=0)
                    embeddings_list.append(emb)
                embeddings = np.stack(embeddings_list, axis=0) # variable length

                # Update stats
                encoding_time = time.time() - start_time
                self.stats["total_embeddings_generated"] += len(images)
                self.stats["avg_encoding_time"] = (
                    self.stats["avg_encoding_time"] * 0.9 + encoding_time * 0.1
                )

                logger.info(
                    f"Encoded {len(images)} images in {encoding_time:.2f}s. "
                    f"Embeddings shape: {embeddings.shape}"
                )

                return embeddings

            except Exception as e:
                logger.error(f"Error encoding images: {traceback.format_exc()}")
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
            embeddings: Image embeddings array (num_images, n_vectors, embedding_dim)
            image_paths: Corresponding image file paths
        """
        try:
            if embeddings.shape[0] == 0:
                raise MRAGError("Cannot build index from empty embeddings")

            if len(image_paths) != embeddings.shape[0]:
                raise MRAGError(
                    f"Mismatch between embeddings ({embeddings.shape[0]}) "
                    f"and image paths ({len(image_paths)})"
                )

            logger.info(f"Building FAISS index for {embeddings.shape[0]} images...")

            with self.memory_manager.memory_guard("FAISS index building"):
                # Ensure embeddings are float32 and normalized
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)
                print(embeddings.shape)

                # Normalize embeddings if not already normalized
                norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
                if not np.allclose(norms, 1.0, atol=1e-3):
                    # Normalize vectors except 0
                    embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=(norms!=0))

                # Flatten embeddings and repeat image paths
                global_embeddings = []
                region_embeddings = []
                region_image_paths = []
                for i in range(embeddings.shape[0]):
                    emb = embeddings[i]
                    global_emb = emb[0]  # First vector is global
                    global_embeddings.append(global_emb)
                    for j in range(1, emb.shape[0]):
                        # Ignore zero vectors
                        if np.linalg.norm(emb[j]) > 0:
                            region_embeddings.append(emb[j])
                            region_image_paths.append(image_paths[i])

                # Choose index type based on corpus size
                global_embeddings = np.stack(global_embeddings, axis=0)
                num_embeddings = global_embeddings.shape[0]

                # Region embeddings are guaranteed to exist
                region_embeddings = np.stack(region_embeddings, axis=0)

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
                    self.index.train(global_embeddings)
                    logger.info("Using IVF FAISS index (IndexIVFFlat)")

                # Add embeddings to index
                self.index.add(global_embeddings)

                # Configure IVF index search parameters
                if isinstance(self.index, faiss.IndexIVFFlat):
                    # Set nprobe to search more clusters for better recall
                    # Use 10% of clusters or at least 10
                    self.index.nprobe = max(10, nlist // 10)
                    logger.info(f"Set IVF nprobe to {self.index.nprobe} (searching {self.index.nprobe}/{nlist} clusters)")

                # Store image paths and embeddings
                self.image_paths = image_paths.copy()
                self.global_embeddings = global_embeddings.copy()
                self.region_embeddings = region_embeddings.copy()
                self.region_image_paths = region_image_paths.copy()

                logger.info(
                    f"FAISS index built successfully. "
                    f"Total vectors: {self.index.ntotal}, "
                    f"Index type: {type(self.index).__name__}"
                )
        except Exception as e:
            logger.error(f"Error building FAISS index: {traceback.format_exc()}")
            traceback.print_exc()
            raise MRAGError("Failed to build FAISS index") from e

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
                    embedding=self.global_embeddings[idx] if self.global_embeddings is not None else None,
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
    def retrieve_by_image(self, query_image: Image.Image, k: Optional[int] = None, n: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve most similar images for an image query (image-to-image retrieval).

        Args:
            query_image: PIL Image to use as query
            k: Number of results to return (defaults to config.top_k)
            n: Average number of candidates for each result to return

        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        if self.index is None:
            raise MRAGError("Index not built. Call build_index() first.")

        if k is None:
            k = self.config.top_k
        
        if n is None:
            n = 20

        start_time = time.time()
        try:
            with self.memory_manager.memory_guard("Image similarity retrieval"):
                # Encode query image
                query_embeddings = self.encode_images([query_image])[0]

                if query_embeddings.shape[0] == 0:
                    logger.warning("Failed to encode query image, returning empty results")
                    return []

                # Stage 1 global image retrieval
                query_embedding, query_region_embeddings = query_embeddings[0], query_embeddings[1:]
                query_embedding = query_embedding.reshape(1, -1)  # 1 x D
                scores, indices = self.index.search(query_embedding, k*n)

                candidates = []            
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    score = float(scores[0][i])

                    # Skip invalid indices
                    if idx < 0 or idx >= len(self.image_paths):
                        continue

                    # Apply similarity threshold
                    if score < self.config.similarity_threshold:
                        continue
                    candidates.append(self.image_paths[idx])
                
                # Stage 2 region image reranking
                mask = np.isin(self.region_image_paths, candidates) # Indexes candidate regions -> global
                sub_region_embeddings = self.region_embeddings[mask] # M x D
                sub_region_image_paths = np.array(self.region_image_paths)[mask] # Indexes global

                # Remove zero vectors (padding) from query region embeddings
                query_region_embeddings = query_region_embeddings[
                    np.linalg.norm(query_region_embeddings, axis=1)>0]

                # Compute MaxSim between query regions and candidate regions; sum over query regions
                sim_matrix = np.dot(query_region_embeddings, sub_region_embeddings.T) # R x M
                max_sim = np.max(sim_matrix, axis=0) # M

                # Top K results
                topk_indices = np.argsort(-max_sim)[:k]  # indices into the sub_region arrays

                # Map region-level scores back to image paths and keep best score per image
                per_image_best = {}
                for idx_sub in topk_indices:
                    img_path = sub_region_image_paths[idx_sub]
                    score_val = float(max_sim[idx_sub])
                    if img_path not in per_image_best or score_val > per_image_best[img_path]:
                        per_image_best[img_path] = score_val

                # Sort images by best region score and take top-k
                sorted_images = sorted(per_image_best.items(), key=lambda x: -x[1])[:k]

                # Build results using a path->index mapping to avoid repeated .index() calls
                path_to_idx = {p: i for i, p in enumerate(self.image_paths)}
                results = []
                for img_path, score_val in sorted_images:
                    global_idx = path_to_idx.get(img_path, None)
                    results.append(
                        RetrievalResult(
                            image_path=img_path,
                            similarity_score=score_val,
                            embedding=self.global_embeddings[global_idx] if (global_idx is not None and self.global_embeddings is not None) else None,
                            metadata={
                                "index": global_idx,
                                "query_type": "image"
                            }
                        )
                    )

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
        except Exception as e:
            logger.error(f"Error during image retrieval: {traceback.format_exc()}")

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
            "region_image_paths": self.region_image_paths,
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
        if self.global_embeddings is not None:
            self.global_embeddings = np.array(self.global_embeddings)
            embeddings_path = index_path.replace('.bin', '_embeddings.npy')
            np.save(embeddings_path, self.global_embeddings)

        if self.region_embeddings is not None:
            self.region_embeddings = np.array(self.region_embeddings)
            embeddings_path = index_path.replace('.bin', '_region_embeddings.npy')
            np.save(embeddings_path, self.region_embeddings)

        logger.info(f"Index saved to {index_path}")

    @handle_errors
    def load_index(self, index_path: str, image_paths: List[str], region_image_paths: List[str]) -> None:
        """
        Load FAISS index from disk.

        Args:
            index_path: Path to index file
            image_paths: Corresponding image file paths
            region_image_paths: Corresponding repeated image file paths
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
                
                saved_region_image_paths = metadata.get("region_image_paths", [])
                if len(saved_region_image_paths) == self.index.ntotal:
                    self.region_image_paths = saved_region_image_paths
                else:
                    self.region_image_paths = region_image_paths
            else:
                self.image_paths = image_paths
                self.region_image_paths = region_image_paths

            # Load embeddings if available
            embeddings_path = index_path.replace('.bin', '_embeddings.npy')
            if os.path.exists(embeddings_path):
                self.global_embeddings = np.load(embeddings_path)
                logger.info(f"Loaded embeddings: {self.global_embeddings.shape}")

            region_embeddings_path = index_path.replace('.bin', '_region_embeddings.npy')
            if os.path.exists(region_embeddings_path):
                self.region_embeddings = np.load(region_embeddings_path)
                logger.info(f"Loaded region embeddings: {self.region_embeddings.shape}")

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
        
        if self.model_sam is not None:
            del self.model_sam
            self.model_sam = None

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
            "embeddings_shape": self.global_embeddings.shape if self.global_embeddings is not None else None,
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