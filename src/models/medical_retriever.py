"""
Medical Knowledge Retriever using FAISS for efficient similarity search
Handles both image and text retrieval for medical multimodal RAG
"""

import os
import pickle
import numpy as np
import faiss
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MedicalKnowledgeRetriever:
    """
    FAISS-based retrieval system for medical knowledge base
    Supports both image and text embeddings
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "flat",
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize medical knowledge retriever
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('cosine', 'l2')
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Initialize indices
        self.image_index = None
        self.text_index = None
        
        # Metadata storage
        self.image_metadata = []
        self.text_metadata = []
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_texts': 0,
            'index_built': False
        }
        
        logger.info(f"Medical retriever initialized with {index_type} index, {metric} metric")
    
    def _create_index(self, embedding_dim: int, use_gpu: bool = False) -> faiss.Index:
        """
        Create FAISS index based on configuration
        
        Args:
            embedding_dim: Dimension of embeddings
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            FAISS index
        """
        if self.metric == "cosine":
            # For cosine similarity, use inner product on normalized vectors
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(embedding_dim)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, embedding_dim, self.nlist)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(embedding_dim, 32)  # M=32 connections
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        else:  # L2 distance
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(embedding_dim)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, embedding_dim, self.nlist)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(embedding_dim, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using GPU acceleration for FAISS")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def add_images(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        image_paths: Optional[List[str]] = None
    ):
        """
        Add image embeddings and metadata to the retriever
        
        Args:
            embeddings: Image embeddings [num_images, embedding_dim]
            metadata: List of metadata dictionaries for each image
            image_paths: Optional list of image file paths
        """
        if len(embeddings.shape) != 2:
            raise ValueError("Embeddings must be 2D array [num_images, embedding_dim]")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("Number of metadata entries must match number of embeddings")
        
        # Initialize image index if not exists
        if self.image_index is None:
            self.image_index = self._create_index(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        
        # Train index if needed
        if hasattr(self.image_index, 'is_trained') and not self.image_index.is_trained:
            logger.info("Training image index...")
            self.image_index.train(embeddings.astype('float32'))
        
        # Add to index
        self.image_index.add(embeddings.astype('float32'))
        
        # Store metadata with additional information
        for i, meta in enumerate(metadata):
            enhanced_meta = meta.copy()
            enhanced_meta['index_id'] = self.stats['total_images'] + i
            enhanced_meta['embedding_norm'] = float(np.linalg.norm(embeddings[i]))
            
            if image_paths:
                enhanced_meta['file_path'] = image_paths[i]
            
            self.image_metadata.append(enhanced_meta)
        
        # Update statistics
        self.stats['total_images'] += embeddings.shape[0]
        logger.info(f"Added {embeddings.shape[0]} images. Total: {self.stats['total_images']}")
    
    def add_texts(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        texts: Optional[List[str]] = None
    ):
        """
        Add text embeddings and metadata to the retriever
        
        Args:
            embeddings: Text embeddings [num_texts, embedding_dim]
            metadata: List of metadata dictionaries for each text
            texts: Optional list of original texts
        """
        if len(embeddings.shape) != 2:
            raise ValueError("Embeddings must be 2D array [num_texts, embedding_dim]")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("Number of metadata entries must match number of embeddings")
        
        # Initialize text index if not exists
        if self.text_index is None:
            self.text_index = self._create_index(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        
        # Train index if needed
        if hasattr(self.text_index, 'is_trained') and not self.text_index.is_trained:
            logger.info("Training text index...")
            self.text_index.train(embeddings.astype('float32'))
        
        # Add to index
        self.text_index.add(embeddings.astype('float32'))
        
        # Store metadata
        for i, meta in enumerate(metadata):
            enhanced_meta = meta.copy()
            enhanced_meta['index_id'] = self.stats['total_texts'] + i
            enhanced_meta['embedding_norm'] = float(np.linalg.norm(embeddings[i]))
            
            if texts:
                enhanced_meta['text'] = texts[i]
            
            self.text_metadata.append(enhanced_meta)
        
        # Update statistics
        self.stats['total_texts'] += embeddings.shape[0]
        logger.info(f"Added {embeddings.shape[0]} texts. Total: {self.stats['total_texts']}")
    
    def search_images(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images
        
        Args:
            query_embedding: Query embedding [embedding_dim] or [1, embedding_dim]
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with metadata and scores
        """
        if self.image_index is None:
            raise ValueError("No images in index. Add images first.")
        
        # Ensure correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query for cosine similarity
        if self.metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 1e-8:
                query_embedding = query_embedding / norm
        
        # Set search parameters for IVF
        if hasattr(self.image_index, 'nprobe'):
            self.image_index.nprobe = self.nprobe
        
        # Perform search
        scores, indices = self.image_index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            metadata = self.image_metadata[idx].copy()
            
            # Apply metadata filters if specified
            if filter_metadata:
                if not self._matches_filter(metadata, filter_metadata):
                    continue
            
            result = {
                'rank': i + 1,
                'score': float(score),
                'metadata': metadata,
                'type': 'image'
            }
            results.append(result)
        
        return results
    
    def search_texts(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts
        
        Args:
            query_embedding: Query embedding [embedding_dim] or [1, embedding_dim]
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with metadata and scores
        """
        if self.text_index is None:
            raise ValueError("No texts in index. Add texts first.")
        
        # Ensure correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query for cosine similarity
        if self.metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 1e-8:
                query_embedding = query_embedding / norm
        
        # Set search parameters for IVF
        if hasattr(self.text_index, 'nprobe'):
            self.text_index.nprobe = self.nprobe
        
        # Perform search
        scores, indices = self.text_index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue
            
            metadata = self.text_metadata[idx].copy()
            
            # Apply metadata filters if specified
            if filter_metadata:
                if not self._matches_filter(metadata, filter_metadata):
                    continue
            
            result = {
                'rank': i + 1,
                'score': float(score),
                'metadata': metadata,
                'type': 'text'
            }
            results.append(result)
        
        return results
    
    def search_multimodal(
        self,
        query_embedding: np.ndarray,
        k_images: int = 3,
        k_texts: int = 2,
        combine_scores: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search both image and text indices
        
        Args:
            query_embedding: Query embedding
            k_images: Number of images to retrieve
            k_texts: Number of texts to retrieve
            combine_scores: Whether to combine and rank all results
            
        Returns:
            Dictionary with image and text results
        """
        results = {
            'images': [],
            'texts': [],
            'combined': []
        }
        
        # Search images
        if self.image_index is not None:
            image_results = self.search_images(query_embedding, k_images)
            results['images'] = image_results
            results['combined'].extend(image_results)
        
        # Search texts
        if self.text_index is not None:
            text_results = self.search_texts(query_embedding, k_texts)
            results['texts'] = text_results
            results['combined'].extend(text_results)
        
        # Sort combined results by score if requested
        if combine_scores and results['combined']:
            results['combined'].sort(key=lambda x: x['score'], reverse=True)
            # Re-rank
            for i, result in enumerate(results['combined']):
                result['rank'] = i + 1
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter criteria
        
        Args:
            metadata: Item metadata
            filter_dict: Filter criteria
            
        Returns:
            True if metadata matches all filter criteria
        """
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        stats = self.stats.copy()
        
        if self.image_index:
            stats['image_index_trained'] = getattr(self.image_index, 'is_trained', True)
            stats['image_index_total'] = self.image_index.ntotal
        
        if self.text_index:
            stats['text_index_trained'] = getattr(self.text_index, 'is_trained', True)
            stats['text_index_total'] = self.text_index.ntotal
        
        return stats
    
    def save_index(self, save_dir: str):
        """
        Save retriever indices and metadata
        
        Args:
            save_dir: Directory to save to
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indices
        if self.image_index:
            faiss.write_index(self.image_index, str(save_path / "image_index.faiss"))
        
        if self.text_index:
            faiss.write_index(self.text_index, str(save_path / "text_index.faiss"))
        
        # Save metadata
        with open(save_path / "image_metadata.pkl", 'wb') as f:
            pickle.dump(self.image_metadata, f)
        
        with open(save_path / "text_metadata.pkl", 'wb') as f:
            pickle.dump(self.text_metadata, f)
        
        # Save configuration
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'stats': self.stats
        }
        
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Retriever saved to {save_dir}")
    
    @classmethod
    def load_index(cls, load_dir: str, use_gpu: bool = False):
        """
        Load retriever from saved files
        
        Args:
            load_dir: Directory to load from
            use_gpu: Whether to use GPU
            
        Returns:
            Loaded retriever instance
        """
        load_path = Path(load_dir)
        
        # Load configuration
        with open(load_path / "config.json", 'r') as f:
            config = json.load(f)
        
        # Create instance
        retriever = cls(
            embedding_dim=config['embedding_dim'],
            index_type=config['index_type'],
            metric=config['metric'],
            nlist=config['nlist'],
            nprobe=config['nprobe']
        )
        
        # Load indices
        if (load_path / "image_index.faiss").exists():
            retriever.image_index = faiss.read_index(str(load_path / "image_index.faiss"))
            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                retriever.image_index = faiss.index_cpu_to_gpu(res, 0, retriever.image_index)
        
        if (load_path / "text_index.faiss").exists():
            retriever.text_index = faiss.read_index(str(load_path / "text_index.faiss"))
            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                retriever.text_index = faiss.index_cpu_to_gpu(res, 0, retriever.text_index)
        
        # Load metadata
        if (load_path / "image_metadata.pkl").exists():
            with open(load_path / "image_metadata.pkl", 'rb') as f:
                retriever.image_metadata = pickle.load(f)
        
        if (load_path / "text_metadata.pkl").exists():
            with open(load_path / "text_metadata.pkl", 'rb') as f:
                retriever.text_metadata = pickle.load(f)
        
        retriever.stats = config.get('stats', retriever.stats)
        
        logger.info(f"Retriever loaded from {load_dir}")
        return retriever


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize retriever
    retriever = MedicalKnowledgeRetriever(embedding_dim=512)
    
    # Create dummy data for testing
    num_images = 100
    num_texts = 50
    embedding_dim = 512
    
    # Generate random embeddings
    image_embeddings = np.random.randn(num_images, embedding_dim).astype(np.float32)
    text_embeddings = np.random.randn(num_texts, embedding_dim).astype(np.float32)
    
    # Create metadata
    image_metadata = [
        {
            'image_id': f"img_{i}",
            'modality': np.random.choice(['chest_xray', 'ct_scan', 'mri']),
            'condition': np.random.choice(['pneumonia', 'normal', 'fracture']),
            'anatomy': np.random.choice(['chest', 'head', 'abdomen'])
        }
        for i in range(num_images)
    ]
    
    text_metadata = [
        {
            'text_id': f"text_{i}",
            'source': 'medical_textbook',
            'topic': np.random.choice(['cardiology', 'radiology', 'pathology'])
        }
        for i in range(num_texts)
    ]
    
    # Add to retriever
    retriever.add_images(image_embeddings, image_metadata)
    retriever.add_texts(text_embeddings, text_metadata)
    
    # Test search
    query_embedding = np.random.randn(embedding_dim).astype(np.float32)
    
    # Search images
    image_results = retriever.search_images(query_embedding, k=5)
    print(f"Found {len(image_results)} image results")
    
    # Search texts
    text_results = retriever.search_texts(query_embedding, k=3)
    print(f"Found {len(text_results)} text results")
    
    # Multimodal search
    multimodal_results = retriever.search_multimodal(query_embedding)
    print(f"Multimodal search: {len(multimodal_results['combined'])} total results")
    
    # Test filtering
    filtered_results = retriever.search_images(
        query_embedding, 
        k=10,
        filter_metadata={'modality': 'chest_xray'}
    )
    print(f"Filtered results (chest X-rays): {len(filtered_results)}")
    
    # Print statistics
    stats = retriever.get_statistics()
    print(f"Statistics: {stats}")
    
    # Test save/load (commented out to avoid file operations in example)
    # retriever.save_index("./test_index")
    # loaded_retriever = MedicalKnowledgeRetriever.load_index("./test_index")