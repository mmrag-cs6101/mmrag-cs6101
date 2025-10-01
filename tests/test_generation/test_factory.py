"""
Unit Tests for Generation Pipeline Factory

Tests for factory pattern implementation and pipeline creation.
"""

import pytest
from unittest.mock import patch, Mock

from src.generation.factory import GenerationPipelineFactory, create_llava_pipeline
from src.generation.interface import GenerationConfig
from src.generation.llava_pipeline import LLaVAGenerationPipeline
from src.utils.error_handling import MRAGError


class TestGenerationPipelineFactory:
    """Test suite for generation pipeline factory."""

    def test_available_pipelines(self):
        """Test available pipeline types."""
        available = GenerationPipelineFactory.get_available_pipelines()

        assert isinstance(available, dict)
        assert "llava" in available
        assert len(available) > 0

    def test_pipeline_type_inference(self):
        """Test pipeline type inference from model names."""
        # Test LLaVA inference
        pipeline_type = GenerationPipelineFactory._infer_pipeline_type("llava-hf/llava-1.5-7b-hf")
        assert pipeline_type == "llava"

        pipeline_type = GenerationPipelineFactory._infer_pipeline_type("LLaVA-1.5-13B")
        assert pipeline_type == "llava"

        # Test unknown model (should default to llava)
        pipeline_type = GenerationPipelineFactory._infer_pipeline_type("unknown-model")
        assert pipeline_type == "llava"

    @patch('src.generation.factory.LLaVAGenerationPipeline')
    def test_create_pipeline_with_explicit_type(self, mock_llava_class):
        """Test pipeline creation with explicit type."""
        mock_pipeline = Mock()
        mock_llava_class.return_value = mock_pipeline

        config = GenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")

        # Create with explicit type
        result = GenerationPipelineFactory.create_pipeline(config, "llava")

        assert result == mock_pipeline
        mock_llava_class.assert_called_once_with(config)

    @patch('src.generation.factory.LLaVAGenerationPipeline')
    def test_create_pipeline_with_inference(self, mock_llava_class):
        """Test pipeline creation with type inference."""
        mock_pipeline = Mock()
        mock_llava_class.return_value = mock_pipeline

        config = GenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")

        # Create without explicit type (should infer)
        result = GenerationPipelineFactory.create_pipeline(config)

        assert result == mock_pipeline
        mock_llava_class.assert_called_once_with(config)

    def test_create_pipeline_unsupported_type(self):
        """Test error handling for unsupported pipeline type."""
        config = GenerationConfig()

        with pytest.raises(MRAGError) as exc_info:
            GenerationPipelineFactory.create_pipeline(config, "unsupported_type")

        assert "Unsupported pipeline type" in str(exc_info.value)
        assert "unsupported_type" in str(exc_info.value)

    @patch('src.generation.factory.LLaVAGenerationPipeline')
    def test_create_pipeline_construction_error(self, mock_llava_class):
        """Test error handling when pipeline construction fails."""
        mock_llava_class.side_effect = Exception("Construction failed")

        config = GenerationConfig()

        with pytest.raises(MRAGError) as exc_info:
            GenerationPipelineFactory.create_pipeline(config, "llava")

        assert "Pipeline creation failed" in str(exc_info.value)

    def test_create_default_config(self):
        """Test default configuration creation."""
        # Test with defaults
        config = GenerationPipelineFactory.create_default_config()

        assert isinstance(config, GenerationConfig)
        assert config.model_name == "llava-hf/llava-1.5-7b-hf"
        assert config.max_memory_gb == 5.0
        assert config.quantization == "4bit"

        # Test with custom parameters
        config = GenerationPipelineFactory.create_default_config(
            model_name="custom-model",
            max_memory_gb=10.0,
            temperature=0.5
        )

        assert config.model_name == "custom-model"
        assert config.max_memory_gb == 10.0
        assert config.temperature == 0.5

    @patch('src.generation.factory.GenerationPipelineFactory.create_pipeline')
    @patch('src.generation.factory.GenerationPipelineFactory.create_default_config')
    def test_create_llava_pipeline_convenience_function(self, mock_create_config, mock_create_pipeline):
        """Test convenience function for creating LLaVA pipeline."""
        mock_config = Mock()
        mock_pipeline = Mock()
        mock_create_config.return_value = mock_config
        mock_create_pipeline.return_value = mock_pipeline

        # Test with defaults
        result = create_llava_pipeline()

        assert result == mock_pipeline
        mock_create_config.assert_called_once_with(
            model_name="llava-hf/llava-1.5-7b-hf",
            max_memory_gb=5.0
        )
        mock_create_pipeline.assert_called_once_with(mock_config, "llava")

        # Reset mocks
        mock_create_config.reset_mock()
        mock_create_pipeline.reset_mock()

        # Test with custom parameters
        result = create_llava_pipeline(
            model_name="custom-llava",
            max_memory_gb=8.0,
            temperature=0.8
        )

        assert result == mock_pipeline
        mock_create_config.assert_called_once_with(
            model_name="custom-llava",
            max_memory_gb=8.0,
            temperature=0.8
        )

    def test_available_pipelines_structure(self):
        """Test structure of available pipelines dictionary."""
        available = GenerationPipelineFactory.get_available_pipelines()

        for pipeline_type, description in available.items():
            assert isinstance(pipeline_type, str)
            assert isinstance(description, str)
            assert len(pipeline_type) > 0
            assert len(description) > 0

    @pytest.mark.parametrize("model_name,expected_type", [
        ("llava-hf/llava-1.5-7b-hf", "llava"),
        ("LLaVA-1.5-13B-hf", "llava"),
        ("llava-next-13b", "llava"),
        ("random-model-name", "llava"),  # Should default to llava
        ("", "llava"),  # Empty string should default to llava
    ])
    def test_pipeline_type_inference_parametrized(self, model_name, expected_type):
        """Parametrized test for pipeline type inference."""
        result = GenerationPipelineFactory._infer_pipeline_type(model_name)
        assert result == expected_type


class TestGenerationConfigValidation:
    """Test configuration validation and edge cases."""

    def test_config_with_extreme_values(self):
        """Test configuration with extreme values."""
        # Very small memory
        config = GenerationPipelineFactory.create_default_config(max_memory_gb=0.1)
        assert config.max_memory_gb == 0.1

        # Very large memory
        config = GenerationPipelineFactory.create_default_config(max_memory_gb=100.0)
        assert config.max_memory_gb == 100.0

        # Zero temperature
        config = GenerationPipelineFactory.create_default_config(temperature=0.0)
        assert config.temperature == 0.0

        # High temperature
        config = GenerationPipelineFactory.create_default_config(temperature=2.0)
        assert config.temperature == 2.0

    def test_config_with_different_devices(self):
        """Test configuration with different device settings."""
        # CPU device
        config = GenerationPipelineFactory.create_default_config(device="cpu")
        assert config.device == "cpu"

        # CUDA device
        config = GenerationPipelineFactory.create_default_config(device="cuda")
        assert config.device == "cuda"

        # Specific CUDA device
        config = GenerationPipelineFactory.create_default_config(device="cuda:1")
        assert config.device == "cuda:1"

    def test_config_with_different_quantization(self):
        """Test configuration with different quantization settings."""
        # 4-bit quantization
        config = GenerationPipelineFactory.create_default_config(quantization="4bit")
        assert config.quantization == "4bit"

        # 8-bit quantization
        config = GenerationPipelineFactory.create_default_config(quantization="8bit")
        assert config.quantization == "8bit"

        # No quantization
        config = GenerationPipelineFactory.create_default_config(quantization="none")
        assert config.quantization == "none"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])