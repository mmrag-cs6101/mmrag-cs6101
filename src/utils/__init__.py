"""
Utilities Module

Provides memory management, VRAM monitoring, system optimization, and error handling utilities.
"""

from .memory_manager import MemoryManager, MemoryMonitor
from .optimization import OptimizationUtils
from .error_handling import (
    ErrorHandler, MRAGError, MemoryError, ModelLoadingError,
    DataProcessingError, InferenceError, ConfigurationError,
    ErrorSeverity, ErrorCategory, with_error_handling, error_context
)

__all__ = [
    "MemoryManager", "MemoryMonitor", "OptimizationUtils",
    "ErrorHandler", "MRAGError", "MemoryError", "ModelLoadingError",
    "DataProcessingError", "InferenceError", "ConfigurationError",
    "ErrorSeverity", "ErrorCategory", "with_error_handling", "error_context"
]