"""
Local RAG-Anything Integration
Replaces OpenAI API calls with local open-source models
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Union
import asyncio

# Add RAG-Anything to path
rag_anything_path = Path(__file__).parent.parent.parent / "RAG-Anything"
if rag_anything_path.exists():
    sys.path.insert(0, str(rag_anything_path))

# Import RAG-Anything components
try:
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from raganything import RAGAnything
    from raganything.config import RAGAnythingConfig
except ImportError as e:
    logging.error(f"Failed to import RAG-Anything components: {e}")
    logging.error("Make sure RAG-Anything is properly installed and accessible")
    raise

# Import our local model wrappers
from .local_llm_wrapper import create_local_llm_func, LocalLLMManager
from .local_vision_wrapper import create_local_vision_func, LocalVisionManager
from .local_embedding_wrapper import create_local_embedding_func, create_medical_embedding_func, LocalEmbeddingManager

logger = logging.getLogger(__name__)


class LocalRAGAnything:
    """
    RAG-Anything with local open-source models
    Replaces all API-based models with local alternatives
    """
    
    def __init__(
        self,
        working_dir: str = "./rag_storage",
        # Model configurations
        llm_model: str = "mistral-7b",
        vision_model: str = "llava-1.5-7b", 
        embedding_model: str = "bge-base",
        # Model settings
        device: str = "auto",
        load_in_4bit: bool = True,
        cache_dir: Optional[str] = None,
        # RAG settings
        rag_config: Optional[RAGAnythingConfig] = None,
        # Medical domain specialization
        medical_mode: bool = False
    ):
        """
        Initialize Local RAG-Anything with open-source models
        
        Args:
            working_dir: Directory for RAG storage
            llm_model: LLM model key or name
            vision_model: Vision model key or name  
            embedding_model: Embedding model key or name
            device: Device to use ("cuda", "cpu", "auto")
            load_in_4bit: Use 4-bit quantization for large models
            cache_dir: Model cache directory
            rag_config: RAG-Anything configuration
            medical_mode: Enable medical domain specialization
        """
        self.working_dir = working_dir
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.cache_dir = cache_dir
        self.medical_mode = medical_mode
        
        # Model managers
        self.llm_manager = LocalLLMManager()
        self.vision_manager = LocalVisionManager()
        self.embedding_manager = LocalEmbeddingManager()
        
        # Model configurations
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.embedding_model = embedding_model
        
        # RAG configuration
        self.rag_config = rag_config or RAGAnythingConfig(working_dir=working_dir)
        
        # Initialize model functions
        self._initialize_model_functions()
        
        # Initialize RAG-Anything instance
        self.rag_anything = None
        
        logger.info(f"LocalRAGAnything initialized with:")
        logger.info(f"  - LLM: {llm_model}")
        logger.info(f"  - Vision: {vision_model}")
        logger.info(f"  - Embedding: {embedding_model}")
        logger.info(f"  - Medical mode: {medical_mode}")
        logger.info(f"  - Device: {device}")
    
    def _initialize_model_functions(self):
        """Initialize local model functions"""
        try:
            # Model configuration
            model_config = {
                "device": self.device,
                "load_in_4bit": self.load_in_4bit,
                "cache_dir": self.cache_dir
            }
            
            logger.info("Loading LLM model...")
            self.llm_func = create_local_llm_func(
                model_key=self.llm_model,
                custom_config=model_config,
                manager=self.llm_manager
            )
            logger.info("✅ LLM model loaded")
            
            logger.info("Loading vision model...")
            self.vision_func = create_local_vision_func(
                model_key=self.vision_model,
                custom_config=model_config,
                manager=self.vision_manager
            )
            logger.info("✅ Vision model loaded")
            
            logger.info("Loading embedding model...")
            if self.medical_mode:
                self.embedding_func, self.embedding_dim = create_medical_embedding_func(
                    model_key=self.embedding_model,
                    custom_config={"device": self.device, "cache_dir": self.cache_dir}
                )
            else:
                self.embedding_func, self.embedding_dim = create_local_embedding_func(
                    model_key=self.embedding_model,
                    custom_config={"device": self.device, "cache_dir": self.cache_dir},
                    manager=self.embedding_manager
                )
            logger.info("✅ Embedding model loaded")
            
        except Exception as e:
            logger.error(f"Failed to initialize model functions: {e}")
            raise
    
    async def initialize(self):
        """Initialize RAG-Anything with local models"""
        try:
            logger.info("Initializing LightRAG with local models...")
            
            # Create LightRAG instance with local models
            lightrag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_dim,
                    max_token_size=8192,
                    func=self.embedding_func
                )
            )
            
            # Initialize storage
            await lightrag.initialize_storages()
            await initialize_pipeline_status()
            
            logger.info("Creating RAG-Anything instance...")
            
            # Create RAG-Anything instance
            self.rag_anything = RAGAnything(
                config=self.rag_config,
                lightrag=lightrag,
                vision_model_func=self.vision_func
            )
            
            logger.info("✅ Local RAG-Anything initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG-Anything: {e}")
            raise
    
    async def insert_file(
        self, 
        file_path: Union[str, Path],
        parse_method: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert a file into the RAG system
        
        Args:
            file_path: Path to the file to insert
            parse_method: Parse method ("auto", "ocr", "txt")
            **kwargs: Additional arguments
            
        Returns:
            Processing results
        """
        if self.rag_anything is None:
            await self.initialize()
        
        try:
            logger.info(f"Processing file: {file_path}")
            
            result = await self.rag_anything.ainsert(
                file_path=str(file_path),
                parse_method=parse_method,
                **kwargs
            )
            
            logger.info(f"✅ File processed successfully: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise
    
    async def insert_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert all supported files from a directory
        
        Args:
            directory_path: Path to directory
            recursive: Process subdirectories recursively
            **kwargs: Additional arguments
            
        Returns:
            Processing results
        """
        if self.rag_anything is None:
            await self.initialize()
        
        try:
            logger.info(f"Processing directory: {directory_path}")
            
            result = await self.rag_anything.abatch_insert(
                dir_path=str(directory_path),
                recursive=recursive,
                **kwargs
            )
            
            logger.info(f"✅ Directory processed successfully: {directory_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process directory {directory_path}: {e}")
            raise
    
    async def query(
        self,
        query: str,
        mode: str = "mix",
        multimodal_content: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """
        Query the RAG system
        
        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            multimodal_content: Optional multimodal content
            **kwargs: Additional query parameters
            
        Returns:
            Query response
        """
        if self.rag_anything is None:
            await self.initialize()
        
        try:
            if multimodal_content:
                response = await self.rag_anything.aquery_with_multimodal(
                    query=query,
                    multimodal_content=multimodal_content,
                    mode=mode,
                    **kwargs
                )
            else:
                response = await self.rag_anything.aquery(
                    query=query,
                    mode=mode,
                    **kwargs
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    def query_sync(
        self,
        query: str,
        mode: str = "mix",
        multimodal_content: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous query method
        
        Args:
            query: Query text
            mode: Query mode
            multimodal_content: Optional multimodal content
            **kwargs: Additional query parameters
            
        Returns:
            Query response
        """
        return asyncio.run(self.query(query, mode, multimodal_content, **kwargs))
    
    def insert_file_sync(
        self,
        file_path: Union[str, Path],
        parse_method: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous file insertion
        
        Args:
            file_path: Path to file
            parse_method: Parse method
            **kwargs: Additional arguments
            
        Returns:
            Processing results
        """
        return asyncio.run(self.insert_file(file_path, parse_method, **kwargs))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "llm_model": self.llm_model,
            "vision_model": self.vision_model,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "medical_mode": self.medical_mode,
            "loaded_llm_models": self.llm_manager.list_models(),
            "loaded_vision_models": self.vision_manager.list_models(),
            "loaded_embedding_models": self.embedding_manager.list_models()
        }
    
    def cleanup(self):
        """Clean up resources and unload models"""
        logger.info("Cleaning up Local RAG-Anything...")
        
        # Unload all models to free memory
        for model_key in self.llm_manager.list_models():
            self.llm_manager.unload_model(model_key)
        
        for model_key in self.vision_manager.list_models():
            self.vision_manager.unload_model(model_key)
        
        for model_key in self.embedding_manager.list_models():
            self.embedding_manager.unload_model(model_key)
        
        logger.info("✅ Cleanup completed")


class MedicalRAGAnything(LocalRAGAnything):
    """
    Medical-specialized RAG-Anything with domain-specific optimizations
    """
    
    def __init__(
        self,
        working_dir: str = "./medical_rag_storage",
        llm_model: str = "mistral-7b",
        vision_model: str = "llava-1.5-7b",
        embedding_model: str = "bge-base",
        **kwargs
    ):
        """
        Initialize Medical RAG-Anything
        
        Args:
            working_dir: Directory for medical RAG storage
            llm_model: LLM model for medical text processing
            vision_model: Vision model for medical images
            embedding_model: Embedding model for medical texts
            **kwargs: Additional arguments
        """
        # Force medical mode
        kwargs['medical_mode'] = True
        
        super().__init__(
            working_dir=working_dir,
            llm_model=llm_model,
            vision_model=vision_model,
            embedding_model=embedding_model,
            **kwargs
        )
        
        logger.info("Medical RAG-Anything initialized with domain specialization")
    
    async def insert_medical_document(
        self,
        file_path: Union[str, Path],
        document_type: str = "general",
        patient_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert medical document with specialized processing
        
        Args:
            file_path: Path to medical document
            document_type: Type of medical document ("radiology", "pathology", "clinical_notes", etc.)
            patient_id: Optional patient identifier
            **kwargs: Additional arguments
            
        Returns:
            Processing results
        """
        # Add medical-specific metadata
        metadata = kwargs.get('metadata', {})
        metadata.update({
            'document_type': document_type,
            'domain': 'medical'
        })
        if patient_id:
            metadata['patient_id'] = patient_id
        
        kwargs['metadata'] = metadata
        
        return await self.insert_file(file_path, **kwargs)


def create_local_rag_anything(
    working_dir: str = "./rag_storage",
    model_preset: str = "balanced",
    medical_mode: bool = False,
    **kwargs
) -> LocalRAGAnything:
    """
    Create Local RAG-Anything with predefined model presets
    
    Args:
        working_dir: Working directory
        model_preset: Model preset ("fast", "balanced", "quality", "medical")
        medical_mode: Enable medical domain specialization
        **kwargs: Additional configuration
        
    Returns:
        LocalRAGAnything instance
    """
    
    # Predefined model presets
    presets = {
        "fast": {
            "llm_model": "phi-3",
            "vision_model": "llava-1.5-7b",
            "embedding_model": "bge-small",
            "load_in_4bit": False
        },
        "balanced": {
            "llm_model": "mistral-7b",
            "vision_model": "llava-1.5-7b", 
            "embedding_model": "bge-base",
            "load_in_4bit": True
        },
        "quality": {
            "llm_model": "llama2-7b",
            "vision_model": "llava-1.5-13b",
            "embedding_model": "bge-large",
            "load_in_4bit": True
        },
        "medical": {
            "llm_model": "mistral-7b",
            "vision_model": "llava-1.5-7b",
            "embedding_model": "bge-base",
            "load_in_4bit": True,
            "medical_mode": True
        }
    }
    
    # Get preset configuration
    if model_preset in presets:
        config = presets[model_preset].copy()
    else:
        logger.warning(f"Unknown preset '{model_preset}', using 'balanced'")
        config = presets["balanced"].copy()
    
    # Override with user kwargs
    config.update(kwargs)
    
    # Force medical mode for medical preset
    if model_preset == "medical" or medical_mode:
        config["medical_mode"] = True
        return MedicalRAGAnything(working_dir=working_dir, **config)
    else:
        return LocalRAGAnything(working_dir=working_dir, **config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_local_rag_anything():
        """Test Local RAG-Anything"""
        
        # Create Local RAG-Anything instance
        local_rag = create_local_rag_anything(
            working_dir="./test_rag_storage",
            model_preset="fast"  # Use fast preset for testing
        )
        
        try:
            # Initialize
            await local_rag.initialize()
            
            # Test query without documents
            response = await local_rag.query(
                "What is artificial intelligence?"
            )
            print(f"Response: {response}")
            
            # Get model info
            info = local_rag.get_model_info()
            print(f"Model info: {info}")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
        finally:
            # Cleanup
            local_rag.cleanup()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    asyncio.run(test_local_rag_anything())