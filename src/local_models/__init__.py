"""
Local Models Package for RAG-Anything Integration
Provides local open-source alternatives to API-based models
"""

from .local_llm_wrapper import (
    LocalLLMWrapper,
    LocalLLMManager,
    create_local_llm_func,
    RECOMMENDED_MODELS
)

from .local_vision_wrapper import (
    LocalVisionWrapper,
    LocalVisionManager,
    create_local_vision_func,
    RECOMMENDED_VISION_MODELS
)

from .local_embedding_wrapper import (
    LocalEmbeddingWrapper,
    LocalEmbeddingManager,
    MedicalEmbeddingWrapper,
    create_local_embedding_func,
    create_medical_embedding_func,
    RECOMMENDED_EMBEDDING_MODELS
)

from .local_rag_anything import (
    LocalRAGAnything,
    MedicalRAGAnything,
    create_local_rag_anything
)

__all__ = [
    # LLM Components
    "LocalLLMWrapper",
    "LocalLLMManager", 
    "create_local_llm_func",
    "RECOMMENDED_MODELS",
    
    # Vision Components
    "LocalVisionWrapper",
    "LocalVisionManager",
    "create_local_vision_func",
    "RECOMMENDED_VISION_MODELS",
    
    # Embedding Components
    "LocalEmbeddingWrapper",
    "LocalEmbeddingManager",
    "MedicalEmbeddingWrapper",
    "create_local_embedding_func",
    "create_medical_embedding_func",
    "RECOMMENDED_EMBEDDING_MODELS",
    
    # RAG Integration
    "LocalRAGAnything",
    "MedicalRAGAnything",
    "create_local_rag_anything"
]

# Version info
__version__ = "0.1.0"
__author__ = "Medical RAG Team"
__description__ = "Local open-source models for RAG-Anything integration"