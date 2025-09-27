"""
Error Handling Tests

Tests for error handling, recovery mechanisms, and system resilience.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.utils.error_handling import (
    ErrorHandler, MRAGError, MemoryError, ModelLoadingError,
    DataProcessingError, InferenceError, ConfigurationError,
    ErrorSeverity, ErrorCategory, ErrorContext,
    with_error_handling, error_context
)


class TestErrorClasses:
    """Test custom error classes."""

    def test_mrag_error_basic(self):
        """Test basic MRAGError functionality."""
        error = MRAGError("Test error message")
        assert str(error) == "Test error message"
        assert error.context is None
        assert error.original_error is None
        assert error.timestamp > 0

    def test_mrag_error_with_context(self):
        """Test MRAGError with context."""
        context = ErrorContext(
            operation="test_op",
            component="test_component",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            metadata={"key": "value"}
        )

        error = MRAGError("Test error", context=context)
        assert error.context == context
        assert error.context.operation == "test_op"

    def test_mrag_error_with_original_error(self):
        """Test MRAGError wrapping original exception."""
        original = ValueError("Original error")
        error = MRAGError("Wrapped error", original_error=original)
        assert error.original_error == original

    def test_specific_error_types(self):
        """Test specific error type inheritance."""
        memory_error = MemoryError("Memory issue")
        model_error = ModelLoadingError("Model issue")
        data_error = DataProcessingError("Data issue")
        inference_error = InferenceError("Inference issue")
        config_error = ConfigurationError("Config issue")

        assert isinstance(memory_error, MRAGError)
        assert isinstance(model_error, MRAGError)
        assert isinstance(data_error, MRAGError)
        assert isinstance(inference_error, MRAGError)
        assert isinstance(config_error, MRAGError)


class TestErrorContext:
    """Test ErrorContext dataclass."""

    def test_error_context_creation(self):
        """Test ErrorContext creation and defaults."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            timestamp=0,  # Will be set in __post_init__
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_PROCESSING,
            metadata={"batch_size": 4}
        )

        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.timestamp > 0  # Should be set automatically
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.category == ErrorCategory.DATA_PROCESSING
        assert context.metadata["batch_size"] == 4


class TestErrorHandler:
    """Test ErrorHandler class."""

    def test_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler(max_retries=5, base_delay=2.0)
        assert handler.max_retries == 5
        assert handler.base_delay == 2.0
        assert len(handler.error_counts) == 0
        assert ErrorCategory.MEMORY in handler.recovery_strategies

    def test_error_count_tracking(self):
        """Test error count tracking."""
        handler = ErrorHandler()

        context = ErrorContext(
            operation="test_op",
            component="test_component",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.MEMORY,
            metadata={}
        )

        # Simulate error count tracking
        handler._update_error_counts(context)
        handler._update_error_counts(context)

        key = f"{context.component}:{context.category.value}"
        assert handler.error_counts[key] == 2

    def test_error_summary(self):
        """Test error summary generation."""
        handler = ErrorHandler()
        handler.error_counts = {
            "component1:memory": 3,
            "component2:inference": 1,
            "component1:data_processing": 2
        }

        summary = handler.get_error_summary()
        assert summary["total_errors"] == 6
        assert len(summary["most_common_errors"]) == 3
        assert summary["most_common_errors"][0] == ("component1:memory", 3)

    def test_memory_recovery_strategies(self):
        """Test memory-related recovery strategies."""
        handler = ErrorHandler()
        context = ErrorContext(
            operation="test_op",
            component="test_component",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            metadata={"batch_size": 8}
        )

        # Test GPU memory clearing
        with patch('src.utils.error_handling.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            result = handler._clear_gpu_memory(Exception("memory error"), context)
            assert result is True
            mock_torch.cuda.empty_cache.assert_called()

        # Test batch size reduction
        result = handler._reduce_batch_size(Exception("memory error"), context)
        assert result == {"batch_size": 4}  # 8 // 2
        assert context.metadata["batch_size"] == 4

        # Test CPU offload enablement
        result = handler._enable_cpu_offload(Exception("memory error"), context)
        assert result == {"use_cpu_offload": True}
        assert context.metadata["use_cpu_offload"] is True

    def test_model_loading_recovery_strategies(self):
        """Test model loading recovery strategies."""
        handler = ErrorHandler()
        context = ErrorContext(
            operation="load_model",
            component="model_loader",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL_LOADING,
            metadata={"retry_count": 0}
        )

        # Test retry mechanism
        result = handler._retry_model_loading(Exception("loading failed"), context)
        assert result == {"retry": True}
        assert context.metadata["retry_count"] == 1

        # Test fallback model usage
        context.metadata["fallback_models"] = ["model1", "model2"]
        result = handler._use_fallback_model(Exception("loading failed"), context)
        assert result == {"model_name": "model1"}
        assert context.metadata["model_name"] == "model1"
        assert "model2" in context.metadata["fallback_models"]

        # Test CPU mode fallback
        result = handler._enable_cpu_mode(Exception("loading failed"), context)
        assert result == {"device": "cpu"}
        assert context.metadata["device"] == "cpu"

    def test_data_processing_recovery_strategies(self):
        """Test data processing recovery strategies."""
        handler = ErrorHandler()
        context = ErrorContext(
            operation="process_data",
            component="data_loader",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_PROCESSING,
            metadata={"data_index": 42}
        )

        # Test data skipping
        result = handler._skip_corrupted_data(Exception("corrupted data"), context)
        assert result == {"skip": True}
        assert 42 in context.metadata["corrupted_indices"]

        # Test default preprocessing fallback
        result = handler._use_default_preprocessing(Exception("preprocessing failed"), context)
        assert result == {"use_default_preprocessing": True}
        assert context.metadata["use_default_preprocessing"] is True

    def test_inference_recovery_strategies(self):
        """Test inference recovery strategies."""
        handler = ErrorHandler()
        context = ErrorContext(
            operation="generate_answer",
            component="inference_engine",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INFERENCE,
            metadata={"max_length": 512}
        )

        # Test sequence length reduction
        result = handler._reduce_sequence_length(Exception("sequence too long"), context)
        assert result == {"max_length": 256}  # 512 // 2
        assert context.metadata["max_length"] == 256

        # Test inference engine restart
        result = handler._restart_inference_engine(Exception("engine error"), context)
        assert result == {"restart_engine": True}
        assert context.metadata["restart_engine"] is True

    def test_handle_error_with_recovery(self):
        """Test error handling with successful recovery."""
        handler = ErrorHandler()

        context = ErrorContext(
            operation="test_op",
            component="test_component",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.MEMORY,
            metadata={"batch_size": 8}
        )

        error = Exception("Test error")

        # Mock successful recovery
        with patch.object(handler, '_reduce_batch_size', return_value={"batch_size": 4}):
            result = handler.handle_error(error, context)
            assert result == {"batch_size": 4}

    def test_handle_error_all_strategies_fail(self):
        """Test error handling when all recovery strategies fail."""
        handler = ErrorHandler()

        context = ErrorContext(
            operation="test_op",
            component="test_component",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            metadata={}
        )

        error = Exception("Unrecoverable error")

        # Mock all strategies returning None (failure)
        with patch.object(handler, '_clear_gpu_memory', return_value=None):
            with patch.object(handler, '_reduce_batch_size', return_value=None):
                with patch.object(handler, '_enable_cpu_offload', return_value=None):
                    with pytest.raises(Exception) as exc_info:
                        handler.handle_error(error, context)
                    assert exc_info.value == error


class TestErrorDecorators:
    """Test error handling decorators."""

    def test_with_error_handling_decorator_success(self):
        """Test error handling decorator with successful function."""
        @with_error_handling(ErrorCategory.MEMORY, ErrorSeverity.MEDIUM, max_retries=2)
        def successful_function(x, y):
            return x + y

        result = successful_function(2, 3)
        assert result == 5

    def test_with_error_handling_decorator_retry(self):
        """Test error handling decorator with retry mechanism."""
        call_count = 0

        @with_error_handling(ErrorCategory.MEMORY, ErrorSeverity.MEDIUM, max_retries=2)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        # Should succeed after 2 retries
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = failing_function()
            assert result == "success"
            assert call_count == 3

    def test_with_error_handling_decorator_max_retries(self):
        """Test error handling decorator reaching max retries."""
        @with_error_handling(ErrorCategory.MEMORY, ErrorSeverity.MEDIUM, max_retries=1)
        def always_failing_function():
            raise ValueError("Always fails")

        with patch('time.sleep'):  # Mock sleep to speed up test
            with patch.object(ErrorHandler, 'handle_error', side_effect=ValueError("Always fails")):
                with pytest.raises(ValueError):
                    always_failing_function()

    def test_error_context_manager(self):
        """Test error context manager."""
        with patch.object(ErrorHandler, 'handle_error') as mock_handle:
            with pytest.raises(ValueError):
                with error_context("test_op", "test_component", ErrorCategory.MEMORY):
                    raise ValueError("Test error")

            mock_handle.assert_called_once()
            args, kwargs = mock_handle.call_args
            error, context = args
            assert isinstance(error, ValueError)
            assert context.operation == "test_op"
            assert context.component == "test_component"
            assert context.category == ErrorCategory.MEMORY

    def test_error_context_manager_success(self):
        """Test error context manager with no errors."""
        with error_context("test_op", "test_component", ErrorCategory.MEMORY):
            result = 2 + 2
            assert result == 4


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""

    def test_complete_error_recovery_workflow(self):
        """Test complete error recovery workflow."""
        handler = ErrorHandler(max_retries=2, base_delay=0.1)

        # Simulate memory error with batch size reduction
        context = ErrorContext(
            operation="batch_processing",
            component="data_loader",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            metadata={"batch_size": 16}
        )

        memory_error = MemoryError("Out of GPU memory")

        # Should recover by reducing batch size
        result = handler.handle_error(memory_error, context)
        assert result["batch_size"] == 8  # Reduced from 16

        # Check error was tracked
        key = f"{context.component}:{context.category.value}"
        assert handler.error_counts[key] == 1

    def test_cascading_recovery_strategies(self):
        """Test cascading recovery strategies."""
        handler = ErrorHandler()

        context = ErrorContext(
            operation="model_inference",
            component="inference_engine",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INFERENCE,
            metadata={"max_length": 1024}
        )

        # Mock first strategy failure, second strategy success
        with patch.object(handler, '_clear_model_cache', return_value=None):
            with patch.object(handler, '_reduce_sequence_length', return_value={"max_length": 512}):
                result = handler.handle_error(Exception("inference error"), context)
                assert result == {"max_length": 512}

    def test_error_handling_with_metadata_preservation(self):
        """Test that error handling preserves and updates metadata correctly."""
        handler = ErrorHandler()

        initial_metadata = {
            "batch_size": 8,
            "device": "cuda",
            "model_name": "test-model"
        }

        context = ErrorContext(
            operation="test_op",
            component="test_component",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.MEMORY,
            metadata=initial_metadata.copy()
        )

        # Recovery should modify batch_size but preserve other metadata
        result = handler._reduce_batch_size(Exception("memory error"), context)

        assert context.metadata["batch_size"] == 4  # Modified
        assert context.metadata["device"] == "cuda"  # Preserved
        assert context.metadata["model_name"] == "test-model"  # Preserved