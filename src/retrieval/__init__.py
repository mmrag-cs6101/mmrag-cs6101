"""
Retrieval Module

Provides CLIP-based image retrieval with FAISS indexing for efficient similarity search.
Implements embedding generation, vector storage, and top-k retrieval.
"""

from .interface import RetrievalPipeline
from .image_retriever import ImageRetriever

__all__ = ["RetrievalPipeline", "ImageRetriever"]