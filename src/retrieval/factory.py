"""
Retrieval Pipeline Factory

Factory class for creating and configuring CLIP-based retrieval pipelines.
Provides easy instantiation with optimized defaults for MRAG-Bench system.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .clip_retriever import CLIPRetriever
from .embedding_processor import EmbeddingProcessor
from .interface import RetrievalConfig
from ..config import MRAGConfig
from ..utils.memory_manager import MemoryManager


logger = logging.getLogger(__name__)


class RetrievalFactory:
    """Factory for creating and configuring retrieval components."""

    @staticmethod
    def create_clip_retriever(
        config: Optional[RetrievalConfig] = None,
        mrag_config: Optional[MRAGConfig] = None
    ) -> CLIPRetriever:
        """
        Create CLIP retriever with optimized configuration.

        Args:
            config: Specific retrieval configuration
            mrag_config: Complete MRAG system configuration

        Returns:
            Configured CLIPRetriever instance
        """
        if config is None:
            if mrag_config is not None:
                # Extract retrieval config from MRAG config
                config = RetrievalConfig(
                    model_name=mrag_config.model.retriever_name,
                    embedding_dim=mrag_config.retrieval.embedding_dim,
                    top_k=mrag_config.retrieval.top_k,
                    similarity_threshold=mrag_config.retrieval.similarity_threshold,
                    batch_size=mrag_config.retrieval.batch_size,
                    max_memory_gb=mrag_config.performance.memory_limit_gb - mrag_config.performance.memory_buffer_gb,
                    device=mrag_config.model.device
                )
            else:
                # Use default configuration optimized for 16GB VRAM
                config = RetrievalConfig(
                    model_name="openai/clip-vit-base-patch32",
                    embedding_dim=512,
                    top_k=5,
                    similarity_threshold=0.0,
                    batch_size=16,
                    max_memory_gb=1.0,  # Conservative for CLIP model
                    device="cuda"
                )

        logger.info(f"Creating CLIP retriever with config: {config}")
        return CLIPRetriever(config)

    @staticmethod
    def create_embedding_processor(
        retriever: CLIPRetriever,
        cache_dir: Optional[str] = None,
        mrag_config: Optional[MRAGConfig] = None
    ) -> EmbeddingProcessor:
        """
        Create embedding processor with memory management.

        Args:
            retriever: CLIPRetriever instance
            cache_dir: Directory for caching embeddings
            mrag_config: Complete MRAG system configuration

        Returns:
            Configured EmbeddingProcessor instance
        """
        if cache_dir is None:
            if mrag_config is not None:
                cache_dir = mrag_config.dataset.embedding_cache_path
            else:
                cache_dir = "data/embeddings"

        # Create memory manager with appropriate limits
        memory_limit = 15.0  # Default
        if mrag_config is not None:
            memory_limit = mrag_config.performance.memory_limit_gb - mrag_config.performance.memory_buffer_gb

        memory_manager = MemoryManager(memory_limit_gb=memory_limit)

        logger.info(f"Creating embedding processor with cache dir: {cache_dir}")
        return EmbeddingProcessor(retriever, memory_manager, cache_dir)

    @staticmethod
    def create_complete_retrieval_system(
        mrag_config: Optional[MRAGConfig] = None,
        force_rebuild_index: bool = False
    ) -> Dict[str, Any]:
        """
        Create complete retrieval system with all components.

        Args:
            mrag_config: Complete MRAG system configuration
            force_rebuild_index: Whether to rebuild index from scratch

        Returns:
            Dictionary containing all retrieval components
        """
        logger.info("Creating complete retrieval system...")

        # Create retriever
        retriever = RetrievalFactory.create_clip_retriever(mrag_config=mrag_config)

        # Create embedding processor
        embedding_processor = RetrievalFactory.create_embedding_processor(
            retriever, mrag_config=mrag_config
        )

        # Determine index path
        if mrag_config is not None:
            index_path = mrag_config.retrieval.index_cache_path
        else:
            index_path = "data/embeddings/faiss_index.bin"

        # Create index directory
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        system = {
            "retriever": retriever,
            "embedding_processor": embedding_processor,
            "index_path": index_path,
            "config": mrag_config,
            "force_rebuild": force_rebuild_index
        }

        logger.info("Complete retrieval system created successfully")
        return system

    @staticmethod
    def get_optimized_config_for_hardware() -> RetrievalConfig:
        """
        Get retrieval configuration optimized for current hardware.

        Returns:
            Hardware-optimized RetrievalConfig
        """
        import torch

        # Detect available hardware
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            # Get GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            if gpu_memory_gb >= 16:
                # High-end GPU configuration
                config = RetrievalConfig(
                    model_name="openai/clip-vit-base-patch32",
                    embedding_dim=512,
                    top_k=10,
                    similarity_threshold=0.0,
                    batch_size=32,
                    max_memory_gb=2.0,
                    device="cuda"
                )
            elif gpu_memory_gb >= 8:
                # Mid-range GPU configuration
                config = RetrievalConfig(
                    model_name="openai/clip-vit-base-patch32",
                    embedding_dim=512,
                    top_k=5,
                    similarity_threshold=0.0,
                    batch_size=16,
                    max_memory_gb=1.0,
                    device="cuda"
                )
            else:
                # Low-end GPU configuration
                config = RetrievalConfig(
                    model_name="openai/clip-vit-base-patch32",
                    embedding_dim=512,
                    top_k=5,
                    similarity_threshold=0.0,
                    batch_size=8,
                    max_memory_gb=0.5,
                    device="cuda"
                )
        else:
            # CPU-only configuration
            config = RetrievalConfig(
                model_name="openai/clip-vit-base-patch32",
                embedding_dim=512,
                top_k=5,
                similarity_threshold=0.0,
                batch_size=4,
                max_memory_gb=0.0,
                device="cpu"
            )

        logger.info(f"Hardware-optimized config for {device}: batch_size={config.batch_size}")
        return config

    @staticmethod
    def validate_system_requirements() -> Dict[str, Any]:
        """
        Validate system requirements for retrieval pipeline.

        Returns:
            Validation results
        """
        validation = {
            "status": "success",
            "requirements_met": True,
            "warnings": [],
            "errors": []
        }

        try:
            # Check PyTorch availability
            import torch
            validation["pytorch_version"] = torch.__version__
            validation["cuda_available"] = torch.cuda.is_available()

            if torch.cuda.is_available():
                validation["gpu_count"] = torch.cuda.device_count()
                validation["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

                # Check minimum GPU memory
                if validation["gpu_memory_gb"] < 4:
                    validation["errors"].append(
                        f"Insufficient GPU memory: {validation['gpu_memory_gb']:.1f}GB < 4GB minimum"
                    )
                elif validation["gpu_memory_gb"] < 8:
                    validation["warnings"].append(
                        f"Limited GPU memory: {validation['gpu_memory_gb']:.1f}GB. Consider reducing batch sizes."
                    )
            else:
                validation["warnings"].append("CUDA not available. Will use CPU (slower performance).")

            # Check required packages
            required_packages = [
                ("transformers", "4.30.0"),
                ("faiss", "1.7.0"),
                ("numpy", "1.21.0"),
                ("pillow", "9.0.0")
            ]

            missing_packages = []
            for package, min_version in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                validation["errors"].append(f"Missing required packages: {missing_packages}")

            # Check disk space (rough estimate)
            import shutil
            disk_usage = shutil.disk_usage(".")
            free_gb = disk_usage.free / 1024**3

            if free_gb < 10:
                validation["errors"].append(f"Insufficient disk space: {free_gb:.1f}GB < 10GB minimum")
            elif free_gb < 50:
                validation["warnings"].append(f"Limited disk space: {free_gb:.1f}GB. Consider freeing space.")

            validation["disk_free_gb"] = free_gb

        except Exception as e:
            validation["errors"].append(f"System validation failed: {str(e)}")

        # Set overall status
        if validation["errors"]:
            validation["status"] = "error"
            validation["requirements_met"] = False
        elif validation["warnings"]:
            validation["status"] = "warning"

        return validation