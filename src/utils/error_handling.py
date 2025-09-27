"""
Error Handling and Recovery Mechanisms

Comprehensive error handling, recovery strategies, and system resilience for MRAG-Bench.
"""

import logging
import traceback
import time
import functools
from typing import Any, Callable, Optional, Dict, List, Type, Union
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    MEMORY = "memory"
    MODEL_LOADING = "model_loading"
    DATA_PROCESSING = "data_processing"
    INFERENCE = "inference"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class MRAGError(Exception):
    """Base exception class for MRAG-Bench specific errors."""

    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.context = context
        self.original_error = original_error
        self.timestamp = time.time()


class MemoryError(MRAGError):
    """Memory-related errors (GPU/CPU memory issues)."""
    pass


class ModelLoadingError(MRAGError):
    """Model loading and initialization errors."""
    pass


class DataProcessingError(MRAGError):
    """Data loading and preprocessing errors."""
    pass


class InferenceError(MRAGError):
    """Model inference and generation errors."""
    pass


class ConfigurationError(MRAGError):
    """Configuration validation and setup errors."""
    pass


class ErrorHandler:
    """Central error handling and recovery system."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize error handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = self._setup_recovery_strategies()

    def _setup_recovery_strategies(self) -> Dict[ErrorCategory, List[Callable]]:
        """Setup recovery strategies for different error categories."""
        return {
            ErrorCategory.MEMORY: [
                self._clear_gpu_memory,
                self._reduce_batch_size,
                self._enable_cpu_offload
            ],
            ErrorCategory.MODEL_LOADING: [
                self._retry_model_loading,
                self._use_fallback_model,
                self._enable_cpu_mode
            ],
            ErrorCategory.DATA_PROCESSING: [
                self._retry_data_operation,
                self._skip_corrupted_data,
                self._use_default_preprocessing
            ],
            ErrorCategory.INFERENCE: [
                self._clear_model_cache,
                self._reduce_sequence_length,
                self._restart_inference_engine
            ]
        }

    def handle_error(self, error: Exception, context: ErrorContext) -> Any:
        """
        Handle error with appropriate recovery strategy.

        Args:
            error: The exception that occurred
            context: Error context information

        Returns:
            Recovery result or re-raises exception if unrecoverable
        """
        self._log_error(error, context)
        self._update_error_counts(context)

        # Attempt recovery based on error category
        recovery_strategies = self.recovery_strategies.get(context.category, [])

        for strategy in recovery_strategies:
            try:
                result = strategy(error, context)
                if result is not None:
                    self.logger.info(f"Recovery successful using {strategy.__name__}")
                    return result
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")

        # If all recovery strategies failed, re-raise the original error
        self.logger.error(f"All recovery strategies failed for {context.operation}")
        raise error

    def _log_error(self, error: Exception, context: ErrorContext) -> None:
        """Log error with full context information."""
        error_info = {
            "operation": context.operation,
            "component": context.component,
            "severity": context.severity.value,
            "category": context.category.value,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "metadata": context.metadata,
            "traceback": traceback.format_exc()
        }

        if context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Error in {context.operation}: {error_info}")
        else:
            self.logger.warning(f"Warning in {context.operation}: {error_info}")

    def _update_error_counts(self, context: ErrorContext) -> None:
        """Update error counts for monitoring."""
        key = f"{context.component}:{context.category.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

    # Recovery Strategies
    def _clear_gpu_memory(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Clear GPU memory to recover from memory errors."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info("GPU memory cleared")
                return True
        except ImportError:
            pass
        return None

    def _reduce_batch_size(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Reduce batch size to handle memory pressure."""
        if "batch_size" in context.metadata:
            current_batch_size = context.metadata["batch_size"]
            new_batch_size = max(1, current_batch_size // 2)
            context.metadata["batch_size"] = new_batch_size
            self.logger.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")
            return {"batch_size": new_batch_size}
        return None

    def _enable_cpu_offload(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Enable CPU offloading for memory-constrained operations."""
        context.metadata["use_cpu_offload"] = True
        self.logger.info("Enabled CPU offloading")
        return {"use_cpu_offload": True}

    def _retry_model_loading(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Retry model loading with different parameters."""
        retry_count = context.metadata.get("retry_count", 0)
        if retry_count < self.max_retries:
            context.metadata["retry_count"] = retry_count + 1
            time.sleep(self.base_delay * (2 ** retry_count))  # Exponential backoff
            self.logger.info(f"Retrying model loading (attempt {retry_count + 1})")
            return {"retry": True}
        return None

    def _use_fallback_model(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Use fallback model configuration."""
        fallback_models = context.metadata.get("fallback_models", [])
        if fallback_models:
            fallback_model = fallback_models.pop(0)
            context.metadata["model_name"] = fallback_model
            self.logger.info(f"Using fallback model: {fallback_model}")
            return {"model_name": fallback_model}
        return None

    def _enable_cpu_mode(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Enable CPU-only mode as fallback."""
        context.metadata["device"] = "cpu"
        self.logger.info("Falling back to CPU-only mode")
        return {"device": "cpu"}

    def _retry_data_operation(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Retry data processing operation."""
        retry_count = context.metadata.get("retry_count", 0)
        if retry_count < self.max_retries:
            context.metadata["retry_count"] = retry_count + 1
            time.sleep(self.base_delay)
            self.logger.info(f"Retrying data operation (attempt {retry_count + 1})")
            return {"retry": True}
        return None

    def _skip_corrupted_data(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Skip corrupted data samples."""
        if "data_index" in context.metadata:
            corrupted_indices = context.metadata.get("corrupted_indices", [])
            corrupted_indices.append(context.metadata["data_index"])
            context.metadata["corrupted_indices"] = corrupted_indices
            self.logger.warning(f"Skipping corrupted data at index {context.metadata['data_index']}")
            return {"skip": True}
        return None

    def _use_default_preprocessing(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Use default preprocessing when custom preprocessing fails."""
        context.metadata["use_default_preprocessing"] = True
        self.logger.info("Using default preprocessing")
        return {"use_default_preprocessing": True}

    def _clear_model_cache(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Clear model inference cache."""
        try:
            import torch
            if hasattr(torch.nn.Module, 'clear_cache'):
                torch.nn.Module.clear_cache()
            self.logger.info("Model cache cleared")
            return True
        except:
            pass
        return None

    def _reduce_sequence_length(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Reduce sequence length for inference."""
        if "max_length" in context.metadata:
            current_length = context.metadata["max_length"]
            new_length = max(64, current_length // 2)
            context.metadata["max_length"] = new_length
            self.logger.info(f"Reduced sequence length from {current_length} to {new_length}")
            return {"max_length": new_length}
        return None

    def _restart_inference_engine(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Restart inference engine."""
        context.metadata["restart_engine"] = True
        self.logger.info("Restarting inference engine")
        return {"restart_engine": True}

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error occurrences."""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "most_common_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# Decorators for automatic error handling
def with_error_handling(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       max_retries: int = 3):
    """
    Decorator for automatic error handling.

    Args:
        category: Error category
        severity: Error severity level
        max_retries: Maximum retry attempts
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler(max_retries=max_retries)

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        operation=func.__name__,
                        component=func.__module__,
                        timestamp=time.time(),
                        severity=severity,
                        category=category,
                        metadata={"attempt": attempt, "args": str(args), "kwargs": str(kwargs)}
                    )

                    if attempt == max_retries:
                        # Last attempt, let handler decide
                        return handler.handle_error(e, context)
                    else:
                        # Retry with backoff
                        time.sleep(handler.base_delay * (2 ** attempt))
                        continue

        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, component: str, category: ErrorCategory,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, **metadata):
    """
    Context manager for error handling.

    Args:
        operation: Name of the operation
        component: Component name
        category: Error category
        severity: Error severity
        **metadata: Additional metadata
    """
    handler = ErrorHandler()
    context = ErrorContext(
        operation=operation,
        component=component,
        timestamp=time.time(),
        severity=severity,
        category=category,
        metadata=metadata
    )

    try:
        yield
    except Exception as e:
        handler.handle_error(e, context)