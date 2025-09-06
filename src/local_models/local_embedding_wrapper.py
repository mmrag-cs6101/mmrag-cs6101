"""
Local Embedding Model Wrapper for RAG-Anything Integration
Replaces OpenAI embedding API calls with local open-source models
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import asyncio
from functools import partial

logger = logging.getLogger(__name__)


class LocalEmbeddingWrapper:
    """
    Local Embedding Model wrapper supporting multiple open-source embedding models
    Compatible with RAG-Anything's embedding function signature
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "auto",
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        use_sentence_transformers: bool = True
    ):
        """
        Initialize local embedding model wrapper
        
        Args:
            model_name: HuggingFace model name (e.g., "BAAI/bge-base-en-v1.5")
            device: Device to use ("cuda", "cpu", or "auto")
            max_seq_length: Maximum sequence length
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Batch size for processing
            cache_dir: Model cache directory
            use_sentence_transformers: Use sentence-transformers library if possible
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.use_sentence_transformers = use_sentence_transformers
        
        # Initialize model
        self._load_model()
        
        logger.info(f"Local Embedding Model initialized: {model_name} on {self.device}")
    
    def _load_model(self):
        """Load embedding model"""
        try:
            if self.use_sentence_transformers:
                # Try to use sentence-transformers first (easier and more optimized)
                try:
                    self.model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=self.cache_dir
                    )
                    self.model.max_seq_length = self.max_seq_length
                    self.embedding_dim = self.model.get_sentence_embedding_dimension()
                    self.model_type = "sentence_transformers"
                    logger.info(f"Loaded model using sentence-transformers: {self.model_name}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load with sentence-transformers: {e}")
                    logger.info("Falling back to transformers library")
            
            # Fallback to transformers library
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            self.model_type = "transformers"
            logger.info(f"Loaded model using transformers: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model {self.model_name}: {e}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for transformers models"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode texts using transformers library"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                model_output = self.model(**encoded)
                embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
                
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _encode_with_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence-transformers library"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Handle single string input
            if isinstance(texts, str):
                texts = [texts]
            
            # Handle empty input
            if not texts:
                return np.array([])
            
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                logger.warning("No valid texts to encode")
                return np.zeros((len(texts), self.embedding_dim))
            
            # Encode based on model type
            if self.model_type == "sentence_transformers":
                embeddings = self._encode_with_sentence_transformers(valid_texts)
            else:
                embeddings = self._encode_with_transformers(valid_texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts) if isinstance(texts, list) else 1, self.embedding_dim))
    
    async def aencode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Async wrapper for encode method
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        # Run encoding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        func = partial(self.encode, texts)
        
        return await loop.run_in_executor(None, func)
    
    def get_rag_compatible_func(self):
        """
        Get RAG-Anything compatible embedding function
        
        Returns:
            Function compatible with LightRAG's embedding_func signature
        """
        async def rag_embedding_func(texts: Union[str, List[str]]) -> np.ndarray:
            """RAG-compatible embedding function"""
            return await self.aencode(texts)
        
        return rag_embedding_func
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize if not already normalized
        if not self.normalize_embeddings:
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        return np.dot(embeddings1, embeddings2.T)


class LocalEmbeddingManager:
    """Manager for multiple local embedding model instances"""
    
    def __init__(self):
        self.models: Dict[str, LocalEmbeddingWrapper] = {}
    
    def load_model(
        self,
        model_name: str,
        nickname: Optional[str] = None,
        **kwargs
    ) -> LocalEmbeddingWrapper:
        """
        Load and cache an embedding model
        
        Args:
            model_name: HuggingFace model name
            nickname: Optional nickname for the model
            **kwargs: Model initialization parameters
            
        Returns:
            LocalEmbeddingWrapper instance
        """
        key = nickname or model_name
        
        if key not in self.models:
            self.models[key] = LocalEmbeddingWrapper(model_name, **kwargs)
        
        return self.models[key]
    
    def get_model(self, key: str) -> Optional[LocalEmbeddingWrapper]:
        """Get model by key"""
        return self.models.get(key)
    
    def list_models(self) -> List[str]:
        """List loaded models"""
        return list(self.models.keys())
    
    def unload_model(self, key: str):
        """Unload model to free memory"""
        if key in self.models:
            del self.models[key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Pre-configured embedding model configurations
RECOMMENDED_EMBEDDING_MODELS = {
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "embedding_dim": 768,
        "max_seq_length": 512
    },
    "bge-large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "embedding_dim": 1024,
        "max_seq_length": 512
    },
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "max_seq_length": 512
    },
    "e5-base": {
        "model_name": "intfloat/e5-base-v2",
        "embedding_dim": 768,
        "max_seq_length": 512
    },
    "e5-large": {
        "model_name": "intfloat/e5-large-v2",
        "embedding_dim": 1024,
        "max_seq_length": 512
    },
    "multilingual-e5-base": {
        "model_name": "intfloat/multilingual-e5-base",
        "embedding_dim": 768,
        "max_seq_length": 512
    },
    "gte-base": {
        "model_name": "thenlper/gte-base",
        "embedding_dim": 768,
        "max_seq_length": 512
    },
    "all-mpnet": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "embedding_dim": 768,
        "max_seq_length": 384
    }
}


def create_local_embedding_func(
    model_key: str = "bge-base",
    custom_config: Optional[Dict[str, Any]] = None,
    manager: Optional[LocalEmbeddingManager] = None
):
    """
    Create a local embedding function for RAG-Anything integration
    
    Args:
        model_key: Key for recommended model or custom model name
        custom_config: Custom model configuration
        manager: Optional model manager instance
        
    Returns:
        RAG-compatible embedding function and embedding dimension
    """
    if manager is None:
        manager = LocalEmbeddingManager()
    
    # Get model configuration
    if model_key in RECOMMENDED_EMBEDDING_MODELS:
        config = RECOMMENDED_EMBEDDING_MODELS[model_key].copy()
    else:
        config = {"model_name": model_key}
    
    if custom_config:
        config.update(custom_config)
    
    # Extract embedding_dim for return
    expected_dim = config.pop("embedding_dim", 768)
    
    # Load model
    model = manager.load_model(nickname=model_key, **config)
    
    # Get actual embedding dimension
    actual_dim = model.get_embedding_dim()
    
    return model.get_rag_compatible_func(), actual_dim


class MedicalEmbeddingWrapper(LocalEmbeddingWrapper):
    """
    Specialized embedding wrapper for medical domain
    Adds medical-specific preprocessing and optimization
    """
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", **kwargs):
        super().__init__(model_name, **kwargs)
        
        # Medical abbreviations and preprocessing
        self.medical_abbreviations = {
            "MI": "myocardial infarction",
            "DM": "diabetes mellitus",
            "HTN": "hypertension",
            "CHF": "congestive heart failure",
            "COPD": "chronic obstructive pulmonary disease",
            "CT": "computed tomography",
            "MRI": "magnetic resonance imaging",
            "ECG": "electrocardiogram",
            "CBC": "complete blood count",
            "BP": "blood pressure"
        }
    
    def _preprocess_medical_text(self, text: str) -> str:
        """Preprocess medical text by expanding abbreviations"""
        processed_text = text
        
        for abbrev, expansion in self.medical_abbreviations.items():
            # Use word boundaries to avoid partial matches
            import re
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            processed_text = re.sub(pattern, expansion, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts with medical preprocessing"""
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess medical texts
        processed_texts = [self._preprocess_medical_text(text) for text in texts]
        
        # Use parent class encoding
        return super().encode(processed_texts)


def create_medical_embedding_func(
    model_key: str = "bge-base",
    custom_config: Optional[Dict[str, Any]] = None
):
    """
    Create a medical-specialized embedding function
    
    Args:
        model_key: Key for recommended model
        custom_config: Custom model configuration
        
    Returns:
        Medical RAG-compatible embedding function and embedding dimension
    """
    # Get model configuration
    if model_key in RECOMMENDED_EMBEDDING_MODELS:
        config = RECOMMENDED_EMBEDDING_MODELS[model_key].copy()
    else:
        config = {"model_name": model_key}
    
    if custom_config:
        config.update(custom_config)
    
    # Extract embedding_dim for return
    expected_dim = config.pop("embedding_dim", 768)
    
    # Create medical embedding wrapper
    model = MedicalEmbeddingWrapper(**config)
    
    # Get actual embedding dimension
    actual_dim = model.get_embedding_dim()
    
    return model.get_rag_compatible_func(), actual_dim


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_local_embedding():
        # Create local embedding function
        embedding_func, embedding_dim = create_local_embedding_func("bge-base")
        
        # Test the function
        texts = [
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks."
        ]
        
        embeddings = await embedding_func(texts)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embedding_dim}")
        
        # Test medical embedding
        medical_func, medical_dim = create_medical_embedding_func("bge-base")
        
        medical_texts = [
            "Patient has MI and DM",
            "ECG shows abnormal patterns",
            "BP is elevated at 180/90"
        ]
        
        medical_embeddings = await medical_func(medical_texts)
        print(f"Medical embeddings shape: {medical_embeddings.shape}")
    
    # Run test
    asyncio.run(test_local_embedding())