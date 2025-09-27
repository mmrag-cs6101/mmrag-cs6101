"""
Configuration System Tests

Tests for YAML configuration loading, validation, and environment variable overrides.
"""

import os
import tempfile
import pytest
from pathlib import Path

from src.config import (
    MRAGConfig, ModelConfig, DatasetConfig, RetrievalConfig,
    GenerationConfig, EvaluationConfig, PerformanceConfig,
    get_default_config, create_default_config_file
)


class TestModelConfig:
    """Test ModelConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.vlm_name == "llava-hf/llava-1.5-7b-hf"
        assert config.retriever_name == "openai/clip-vit-base-patch32"
        assert config.quantization == "4bit"
        assert config.max_memory_gb == 14.0
        assert config.device == "cuda"
        assert config.torch_dtype == "float16"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            vlm_name="custom-model",
            quantization="8bit",
            max_memory_gb=8.0,
            device="cpu"
        )
        assert config.vlm_name == "custom-model"
        assert config.quantization == "8bit"
        assert config.max_memory_gb == 8.0
        assert config.device == "cpu"


class TestMRAGConfig:
    """Test complete MRAG configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        assert isinstance(config, MRAGConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert isinstance(config.performance, PerformanceConfig)

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = get_default_config()
        # Should not raise any exceptions
        assert config.validate() is True

    def test_config_validation_memory_failure(self):
        """Test configuration validation failure for memory constraints."""
        config = get_default_config()
        config.model.max_memory_gb = 20.0  # Exceeds 16GB limit
        config.performance.memory_limit_gb = 16.0
        config.performance.memory_buffer_gb = 1.0

        with pytest.raises(ValueError, match="Model memory requirement"):
            config.validate()

    def test_config_validation_invalid_quantization(self):
        """Test configuration validation failure for invalid quantization."""
        config = get_default_config()
        config.model.quantization = "invalid"

        with pytest.raises(ValueError, match="Invalid quantization"):
            config.validate()

    def test_config_validation_invalid_device(self):
        """Test configuration validation failure for invalid device."""
        config = get_default_config()
        config.model.device = "invalid"

        with pytest.raises(ValueError, match="Invalid device"):
            config.validate()

    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = get_default_config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "dataset" in config_dict
        assert "retrieval" in config_dict
        assert "generation" in config_dict
        assert "evaluation" in config_dict
        assert "performance" in config_dict

        # Check nested structure
        assert "vlm_name" in config_dict["model"]
        assert "batch_size" in config_dict["dataset"]


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_save_and_load_config(self):
        """Test saving and loading configuration from YAML file."""
        config = get_default_config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            # Save configuration
            config.save(config_path)
            assert Path(config_path).exists()

            # Load configuration
            loaded_config = MRAGConfig.load(config_path)

            # Verify loaded configuration matches original
            assert loaded_config.model.vlm_name == config.model.vlm_name
            assert loaded_config.dataset.batch_size == config.dataset.batch_size
            assert loaded_config.retrieval.top_k == config.retrieval.top_k
            assert loaded_config.generation.max_length == config.generation.max_length

        finally:
            # Cleanup
            if Path(config_path).exists():
                os.unlink(config_path)

    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model": {
                "vlm_name": "test-model",
                "quantization": "8bit",
                "max_memory_gb": 8.0
            },
            "dataset": {
                "batch_size": 2,
                "data_path": "test/data"
            },
            "retrieval": {
                "top_k": 3
            }
        }

        config = MRAGConfig.from_dict(config_dict)

        assert config.model.vlm_name == "test-model"
        assert config.model.quantization == "8bit"
        assert config.model.max_memory_gb == 8.0
        assert config.dataset.batch_size == 2
        assert config.dataset.data_path == "test/data"
        assert config.retrieval.top_k == 3

    def test_create_default_config_file(self):
        """Test creation of default configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")

            create_default_config_file(config_path)

            assert Path(config_path).exists()

            # Load and verify the created config
            config = MRAGConfig.load(config_path)
            assert isinstance(config, MRAGConfig)
            assert config.validate() is True


class TestEnvironmentOverrides:
    """Test environment variable overrides."""

    def test_env_override_basic(self, monkeypatch):
        """Test basic environment variable override."""
        # Create a temporary config file
        config = get_default_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        config.save(config_path)

        try:
            # Set environment variables
            monkeypatch.setenv("MRAG_MODEL_MAX_MEMORY_GB", "8.0")
            monkeypatch.setenv("MRAG_DATASET_BATCH_SIZE", "2")

            # Load with environment overrides
            loaded_config = MRAGConfig.load_with_env_override(config_path)

            # Check overrides were applied
            assert loaded_config.model.max_memory_gb == 8.0
            assert loaded_config.dataset.batch_size == 2

        finally:
            if Path(config_path).exists():
                os.unlink(config_path)


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")

            # Step 1: Create default configuration
            create_default_config_file(config_path)

            # Step 2: Load and modify configuration
            config = MRAGConfig.load(config_path)
            config.model.max_memory_gb = 12.0
            config.dataset.batch_size = 8

            # Step 3: Save modified configuration
            config.save(config_path)

            # Step 4: Reload and verify changes
            reloaded_config = MRAGConfig.load(config_path)
            assert reloaded_config.model.max_memory_gb == 12.0
            assert reloaded_config.dataset.batch_size == 8

            # Step 5: Validate configuration
            assert reloaded_config.validate() is True

    def test_directory_creation(self):
        """Test that configuration validation creates required directories."""
        config = get_default_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set paths to temp directory
            config.dataset.data_path = os.path.join(temp_dir, "data")
            config.dataset.embedding_cache_path = os.path.join(temp_dir, "embeddings")
            config.evaluation.results_path = os.path.join(temp_dir, "results", "eval.json")

            # Validate should create directories
            config.validate()

            # Check directories were created
            assert Path(config.dataset.data_path).exists()
            assert Path(config.dataset.embedding_cache_path).exists()
            assert Path(config.evaluation.results_path).parent.exists()