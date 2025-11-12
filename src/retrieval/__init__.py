"""
Retrieval Module

Provides CLIP-based image retrieval with FAISS indexing for efficient similarity search.
Implements embedding generation, vector storage, and top-k retrieval.
"""

from .interface import RetrievalPipeline, RetrievalResult, RetrievalConfig
from .clip_retriever import CLIPRetriever
from .sam_retriever import SAMRetriever
from .embedding_processor import EmbeddingProcessor
from .factory import RetrievalFactory

__all__ = [
    "RetrievalPipeline",
    "RetrievalResult",
    "RetrievalConfig",
    "CLIPRetriever",
    "SAMRetriever",
    "EmbeddingProcessor",
    "RetrievalFactory"
]