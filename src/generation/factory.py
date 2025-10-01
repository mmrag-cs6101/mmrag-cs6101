"""
Generation Pipeline Factory

Factory for creating generation pipeline instances with different model configurations.
"""

import logging
from typing import Optional, Dict, Any

from .interface import GenerationPipeline, GenerationConfig
from .llava_pipeline import LLaVAGenerationPipeline
from ..utils.error_handling import handle_errors, MRAGError


logger = logging.getLogger(__name__)


class GenerationPipelineFactory:
    """Factory for creating generation pipeline instances."""

    # Available pipeline implementations
    AVAILABLE_PIPELINES = {
        "llava": LLaVAGenerationPipeline,
        "llava-1.5-7b": LLaVAGenerationPipeline,
    }

    @staticmethod
    @handle_errors
    def create_pipeline(
        config: GenerationConfig,
        pipeline_type: Optional[str] = None
    ) -> GenerationPipeline:
        """
        Create generation pipeline instance.

        Args:
            config: Generation configuration
            pipeline_type: Type of pipeline to create. If None, infers from model name.

        Returns:
            Configured generation pipeline instance

        Raises:
            MRAGError: If pipeline type is not supported
        """
        # Infer pipeline type from model name if not specified
        if pipeline_type is None:
            pipeline_type = GenerationPipelineFactory._infer_pipeline_type(config.model_name)

        # Validate pipeline type
        if pipeline_type not in GenerationPipelineFactory.AVAILABLE_PIPELINES:
            available = list(GenerationPipelineFactory.AVAILABLE_PIPELINES.keys())
            raise MRAGError(
                f"Unsupported pipeline type: {pipeline_type}. "
                f"Available types: {available}"
            )

        # Create pipeline instance
        pipeline_class = GenerationPipelineFactory.AVAILABLE_PIPELINES[pipeline_type]

        logger.info(f"Creating {pipeline_type} generation pipeline")

        try:
            pipeline = pipeline_class(config)
            logger.info(f"Successfully created {pipeline_type} pipeline")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to create {pipeline_type} pipeline: {e}")
            raise MRAGError(f"Pipeline creation failed: {str(e)}") from e

    @staticmethod
    def _infer_pipeline_type(model_name: str) -> str:
        """
        Infer pipeline type from model name.

        Args:
            model_name: Name of the model

        Returns:
            Inferred pipeline type
        """
        model_name_lower = model_name.lower()

        if "llava" in model_name_lower:
            return "llava"
        else:
            # Default to LLaVA for now
            logger.warning(f"Unknown model name: {model_name}, defaulting to llava pipeline")
            return "llava"

    @staticmethod
    def get_available_pipelines() -> Dict[str, str]:
        """
        Get available pipeline types and their descriptions.

        Returns:
            Dictionary mapping pipeline types to descriptions
        """
        return {
            "llava": "LLaVA-1.5-7B with 4-bit quantization for medical image QA",
        }

    @staticmethod
    @handle_errors
    def create_default_config(
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        max_memory_gb: float = 5.0,
        **kwargs
    ) -> GenerationConfig:
        """
        Create default generation configuration.

        Args:
            model_name: Name of the model to use
            max_memory_gb: Maximum memory limit in GB
            **kwargs: Additional configuration parameters

        Returns:
            Default generation configuration
        """
        config_params = {
            "model_name": model_name,
            "max_memory_gb": max_memory_gb,
            "max_length": 512,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "quantization": "4bit",
            "device": "cuda",
            "torch_dtype": "float16"
        }

        # Override with any provided kwargs
        config_params.update(kwargs)

        logger.debug(f"Creating default config with parameters: {config_params}")

        return GenerationConfig(**config_params)


# Convenience function for quick pipeline creation
def create_llava_pipeline(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    max_memory_gb: float = 5.0,
    **kwargs
) -> LLaVAGenerationPipeline:
    """
    Convenience function to create LLaVA pipeline with default settings.

    Args:
        model_name: LLaVA model name
        max_memory_gb: Memory limit in GB
        **kwargs: Additional configuration parameters

    Returns:
        Configured LLaVA pipeline
    """
    config = GenerationPipelineFactory.create_default_config(
        model_name=model_name,
        max_memory_gb=max_memory_gb,
        **kwargs
    )

    return GenerationPipelineFactory.create_pipeline(config, "llava")