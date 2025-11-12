"""
Optimization Utilities

Performance optimization utilities for MRAG-Bench system.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from functools import wraps

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OptimizationUtils:
    """Performance optimization utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def configure_torch_optimizations() -> None:
        """Configure PyTorch for optimal performance."""
        if not TORCH_AVAILABLE:
            return

        # Enable cuDNN benchmark for consistent input sizes
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Enable TensorFloat-32 for faster training on Ampere
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True

        # Set memory allocation strategy
        if torch.cuda.is_available():
            # Use memory pool for faster allocation
            torch.cuda.memory._set_allocator_settings("backend:native")

    @staticmethod
    def get_quantization_config(quantization_type: str = "4bit") -> Optional[Dict[str, Any]]:
        """
        Get quantization configuration for models.

        Args:
            quantization_type: Type of quantization ("4bit", "8bit", or "none")

        Returns:
            Quantization configuration dictionary or None
        """
        if quantization_type == "none":
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logging.warning("BitsAndBytes not available, quantization disabled")
            return None

        if quantization_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if TORCH_AVAILABLE else "float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization_type == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")

    @staticmethod
    def optimize_model_loading(device: str = "cuda") -> Dict[str, Any]:
        """
        Get optimized model loading parameters.

        Args:
            device: Target device for model

        Returns:
            Dictionary of optimized loading parameters
        """
        params = {
            "torch_dtype": torch.float16 if TORCH_AVAILABLE else "float16",
            "device_map": "auto" if device == "cuda" else None,
            "low_cpu_mem_usage": True,
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            params["attn_implementation"] = "flash_attention_2"

        return params

    @contextmanager
    def performance_context(self, operation_name: str = "operation"):
        """
        Context manager for performance monitoring.

        Args:
            operation_name: Name of the operation being monitored
        """
        start_time = time.time()
        self.logger.debug(f"Starting {operation_name}")

        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Completed {operation_name} in {elapsed_time:.2f}s")

    def timing_decorator(self, operation_name: Optional[str] = None):
        """
        Decorator for timing function execution.

        Args:
            operation_name: Custom name for the operation
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or func.__name__
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_time = time.time() - start_time
                    self.logger.debug(f"{name} completed in {elapsed_time:.2f}s")
            return wrapper
        return decorator

    @staticmethod
    def optimize_batch_processing(total_items: int, max_batch_size: int,
                                 memory_constraint_gb: float) -> int:
        """
        Calculate optimal batch size for processing.

        Args:
            total_items: Total number of items to process
            max_batch_size: Maximum desired batch size
            memory_constraint_gb: Available memory in GB

        Returns:
            Optimized batch size
        """
        # Simple heuristic: reduce batch size if memory constrained
        if memory_constraint_gb < 2.0:
            return min(max_batch_size, 1)
        elif memory_constraint_gb < 4.0:
            return min(max_batch_size, 2)
        elif memory_constraint_gb < 8.0:
            return min(max_batch_size, 4)
        else:
            return max_batch_size

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get detailed device information for optimization.

        Returns:
            Dictionary with device capabilities and recommendations
        """
        info = {
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": False,
            "device_count": 0,
            "current_device": "cpu",
            "recommendations": []
        }

        if TORCH_AVAILABLE:
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["device_count"] = torch.cuda.device_count()
                info["current_device"] = torch.cuda.current_device()

                # Get GPU properties
                props = torch.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["gpu_memory_gb"] = props.total_memory / 1024**3
                info["compute_capability"] = f"{props.major}.{props.minor}"

                # Recommendations based on GPU
                if props.total_memory < 8 * 1024**3:  # Less than 8GB
                    info["recommendations"].append("Use aggressive quantization")
                    info["recommendations"].append("Reduce batch sizes")
                elif props.total_memory < 16 * 1024**3:  # Less than 16GB
                    info["recommendations"].append("Use 4-bit quantization")
                    info["recommendations"].append("Monitor memory usage")
                else:
                    info["recommendations"].append("Standard configuration suitable")

        return info

    @staticmethod
    def configure_deterministic_execution(seed: int = 42) -> None:
        """
        Configure deterministic execution for reproducible results.

        Args:
            seed: Random seed for reproducibility
        """
        import random
        import os
        import numpy as np

        # Set Python random seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch random seeds
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Make CuDNN deterministic
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # Set environment variables for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    @staticmethod
    def estimate_memory_requirements(model_name: str, quantization: str = "4bit") -> Dict[str, float]:
        """
        Estimate memory requirements for model loading.

        Args:
            model_name: Name of the model
            quantization: Quantization type

        Returns:
            Dictionary with estimated memory requirements in GB
        """
        # Rough estimates based on model size
        model_size_estimates = {
            "llava-1.5-7b": {"base": 14.0, "4bit": 4.5, "8bit": 7.0},
            "clip-vit-base": {"base": 0.5, "4bit": 0.5, "8bit": 0.5},
        }

        # Extract model type from name
        model_type = None
        for key in model_size_estimates:
            if key in model_name.lower():
                model_type = key
                break

        if model_type is None:
            # Default estimate for unknown models
            return {"model_memory": 8.0, "working_memory": 2.0, "total_estimated": 10.0}

        estimates = model_size_estimates[model_type]
        model_memory = estimates.get(quantization, estimates["base"])
        working_memory = model_memory * 0.3  # 30% overhead for working memory

        return {
            "model_memory": model_memory,
            "working_memory": working_memory,
            "total_estimated": model_memory + working_memory
        }