"""
Configuration Management System

YAML-based configuration with validation and environment variable support.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    vlm_name: str = "llava-hf/llava-1.5-7b-hf"
    retriever_name: str = "openai/clip-vit-base-patch32"
    quantization: str = "4bit"
    max_memory_gb: float = 14.0
    device: str = "cuda"
    torch_dtype: str = "float16"


@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""
    data_path: str = "data/mrag_bench"
    batch_size: int = 4
    image_size: tuple = (224, 224)
    cache_embeddings: bool = True
    embedding_cache_path: str = "data/embeddings"


@dataclass
class RetrievalConfig:
    """Retrieval configuration parameters."""
    embedding_dim: int = 512
    top_k: int = 5
    similarity_threshold: float = 0.0
    batch_size: int = 16
    faiss_index_type: str = "IVF"
    index_cache_path: str = "data/embeddings/faiss_index.bin"


@dataclass
class GenerationConfig:
    """Generation configuration parameters."""
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    scenarios: list = field(default_factory=lambda: ["angle", "partial", "scope", "occlusion"])
    save_results: bool = True
    results_path: str = "results/evaluation_results.json"
    log_failed_questions: bool = True


@dataclass
class PerformanceConfig:
    """Performance and resource constraints."""
    memory_limit_gb: float = 16.0
    memory_buffer_gb: float = 1.0
    retrieval_timeout: float = 5.0
    generation_timeout: float = 25.0
    total_pipeline_timeout: float = 30.0
    enable_memory_monitoring: bool = True


@dataclass
class MRAGConfig:
    """Complete MRAG-Bench system configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid, raises ValueError otherwise
        """
        # Validate memory constraints
        total_memory = self.model.max_memory_gb
        if total_memory > (self.performance.memory_limit_gb - self.performance.memory_buffer_gb):
            raise ValueError(
                f"Model memory requirement ({total_memory}GB) exceeds available memory "
                f"({self.performance.memory_limit_gb - self.performance.memory_buffer_gb}GB)"
            )

        # Validate paths
        data_path = Path(self.dataset.data_path)
        if not data_path.exists():
            os.makedirs(data_path, exist_ok=True)

        embedding_path = Path(self.dataset.embedding_cache_path)
        if not embedding_path.exists():
            os.makedirs(embedding_path, exist_ok=True)

        results_path = Path(self.evaluation.results_path).parent
        if not results_path.exists():
            os.makedirs(results_path, exist_ok=True)

        # Validate model configuration
        if self.model.quantization not in ["4bit", "8bit", "none"]:
            raise ValueError(f"Invalid quantization: {self.model.quantization}")

        if self.model.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.model.device}")

        # Validate retrieval configuration
        if self.retrieval.top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.retrieval.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MRAGConfig':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            retrieval=RetrievalConfig(**config_dict.get("retrieval", {})),
            generation=GenerationConfig(**config_dict.get("generation", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            performance=PerformanceConfig(**config_dict.get("performance", {}))
        )

    @classmethod
    def load(cls, filepath: str) -> 'MRAGConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def load_with_env_override(cls, filepath: str, env_prefix: str = "MRAG_") -> 'MRAGConfig':
        """
        Load configuration with environment variable overrides.

        Args:
            filepath: Path to YAML configuration file
            env_prefix: Prefix for environment variables

        Returns:
            MRAGConfig with environment overrides applied
        """
        config = cls.load(filepath)

        # Override with environment variables
        env_overrides = {}
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert MRAG_MODEL_MAX_MEMORY_GB to model.max_memory_gb
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                env_overrides[config_key] = value

        # Apply overrides (simplified implementation)
        if "model.max_memory_gb" in env_overrides:
            config.model.max_memory_gb = float(env_overrides["model.max_memory_gb"])
        if "performance.memory_limit_gb" in env_overrides:
            config.performance.memory_limit_gb = float(env_overrides["performance.memory_limit_gb"])
        if "dataset.batch_size" in env_overrides:
            config.dataset.batch_size = int(env_overrides["dataset.batch_size"])

        return config


def get_default_config() -> MRAGConfig:
    """Get default configuration for MRAG-Bench reproduction."""
    return MRAGConfig()


def create_default_config_file(filepath: str = "config/mrag_bench.yaml") -> None:
    """Create default configuration file."""
    config = get_default_config()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    config.save(filepath)