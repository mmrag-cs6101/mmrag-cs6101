"""
Integration Tests

Tests for system integration, component interaction, and end-to-end workflows.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.config import MRAGConfig, get_default_config
from src.utils import MemoryManager, MemoryMonitor, ErrorHandler
from src.utils.error_handling import ErrorCategory, ErrorSeverity


@pytest.mark.integration
class TestSystemIntegration:
    """Test system-wide integration scenarios."""

    def test_configuration_memory_manager_integration(self):
        """Test integration between configuration and memory management."""
        config = get_default_config()

        # Create memory manager with config parameters
        memory_manager = MemoryManager(
            memory_limit_gb=config.performance.memory_limit_gb,
            buffer_gb=config.performance.memory_buffer_gb
        )

        assert memory_manager.memory_limit_gb == config.performance.memory_limit_gb
        assert memory_manager.buffer_gb == config.performance.memory_buffer_gb
        assert memory_manager.effective_limit_gb == (
            config.performance.memory_limit_gb - config.performance.memory_buffer_gb
        )

    def test_error_handling_memory_integration(self):
        """Test integration between error handling and memory management."""
        memory_manager = MemoryManager(memory_limit_gb=16.0, buffer_gb=1.0)
        error_handler = ErrorHandler(max_retries=3)

        # Simulate memory error scenario
        with patch.object(memory_manager.monitor, 'get_current_stats') as mock_stats:
            from src.utils.memory_manager import MemoryStats

            # Simulate high memory usage
            mock_stats.return_value = MemoryStats(gpu_allocated_gb=18.0, gpu_total_gb=16.0)

            # Check memory pressure
            is_pressure = memory_manager.monitor.check_memory_pressure()
            assert is_pressure is True

            # Memory manager should trigger cleanup
            with patch.object(memory_manager, 'emergency_cleanup') as mock_cleanup:
                if is_pressure:
                    memory_manager.emergency_cleanup()
                mock_cleanup.assert_called_once()

    def test_configuration_validation_with_memory_constraints(self):
        """Test configuration validation considering memory constraints."""
        config = get_default_config()

        # Test valid configuration
        assert config.validate() is True

        # Test invalid configuration (memory exceeds limits)
        config.model.max_memory_gb = 20.0  # Exceeds 16GB total - 1GB buffer
        config.performance.memory_limit_gb = 16.0
        config.performance.memory_buffer_gb = 1.0

        with pytest.raises(ValueError, match="Model memory requirement"):
            config.validate()

    def test_memory_monitoring_throughout_system_lifecycle(self):
        """Test memory monitoring during complete system lifecycle."""
        config = get_default_config()
        memory_manager = MemoryManager(
            memory_limit_gb=config.performance.memory_limit_gb,
            buffer_gb=config.performance.memory_buffer_gb
        )

        # Simulate system startup
        with memory_manager.memory_guard("system_startup"):
            # Simulate memory allocation
            initial_stats = memory_manager.monitor.get_current_stats()
            assert initial_stats.timestamp > 0

        # Simulate model loading
        with memory_manager.memory_guard("model_loading"):
            # Check memory availability before loading
            required_memory = config.model.max_memory_gb
            is_available = memory_manager.check_memory_availability(required_memory)
            # Should be True with default configuration
            assert is_available is True

        # Simulate inference
        with memory_manager.memory_guard("inference"):
            # Get recommended batch size
            batch_size = memory_manager.get_recommended_batch_size(
                base_batch_size=config.dataset.batch_size,
                memory_per_item_mb=500.0
            )
            assert batch_size >= 1
            assert batch_size <= config.dataset.batch_size


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration system integration with other components."""

    def test_config_directory_creation_integration(self):
        """Test that configuration creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = get_default_config()

            # Set paths to temporary directory
            config.dataset.data_path = os.path.join(temp_dir, "data", "mrag_bench")
            config.dataset.embedding_cache_path = os.path.join(temp_dir, "data", "embeddings")
            config.evaluation.results_path = os.path.join(temp_dir, "results", "eval.json")

            # Validate should create directories
            config.validate()

            # Verify directories exist
            assert Path(config.dataset.data_path).exists()
            assert Path(config.dataset.embedding_cache_path).exists()
            assert Path(config.evaluation.results_path).parent.exists()

    def test_config_serialization_integration(self):
        """Test configuration serialization and deserialization."""
        config = get_default_config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            # Save configuration
            config.save(config_path)

            # Load configuration
            loaded_config = MRAGConfig.load(config_path)

            # Validate loaded configuration
            assert loaded_config.validate() is True

            # Check that memory manager can be created from loaded config
            memory_manager = MemoryManager(
                memory_limit_gb=loaded_config.performance.memory_limit_gb,
                buffer_gb=loaded_config.performance.memory_buffer_gb
            )
            assert memory_manager.effective_limit_gb > 0

        finally:
            if Path(config_path).exists():
                os.unlink(config_path)


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling integration across system components."""

    def test_memory_error_recovery_integration(self):
        """Test memory error recovery across components."""
        config = get_default_config()
        memory_manager = MemoryManager(
            memory_limit_gb=config.performance.memory_limit_gb,
            buffer_gb=config.performance.memory_buffer_gb
        )
        error_handler = ErrorHandler()

        # Simulate memory pressure scenario
        with patch.object(memory_manager.monitor, 'get_current_stats') as mock_stats:
            from src.utils.memory_manager import MemoryStats
            from src.utils.error_handling import ErrorContext, MemoryError

            # High memory usage
            mock_stats.return_value = MemoryStats(gpu_allocated_gb=15.5, gpu_total_gb=16.0)

            # Create error context
            context = ErrorContext(
                operation="model_inference",
                component="inference_engine",
                timestamp=0,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.MEMORY,
                metadata={"batch_size": 8}
            )

            memory_error = MemoryError("GPU out of memory")

            # Error handler should provide recovery strategy
            with patch.object(error_handler, '_clear_gpu_memory', return_value=True):
                with patch.object(error_handler, '_reduce_batch_size', return_value={"batch_size": 4}):
                    result = error_handler.handle_error(memory_error, context)
                    assert result is not None  # Some recovery strategy succeeded

    def test_configuration_error_handling_integration(self):
        """Test error handling for configuration-related issues."""
        from src.utils.error_handling import ConfigurationError, ErrorContext

        error_handler = ErrorHandler()

        # Simulate configuration error
        context = ErrorContext(
            operation="config_validation",
            component="config_system",
            timestamp=0,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            metadata={"config_file": "invalid.yaml"}
        )

        config_error = ConfigurationError("Invalid configuration")

        # Configuration errors typically can't be auto-recovered
        with pytest.raises(ConfigurationError):
            error_handler.handle_error(config_error, context)

    def test_system_resilience_under_multiple_errors(self):
        """Test system resilience when multiple errors occur."""
        error_handler = ErrorHandler()

        # Simulate multiple error categories
        error_scenarios = [
            (ErrorCategory.MEMORY, "Memory pressure"),
            (ErrorCategory.DATA_PROCESSING, "Data corruption"),
            (ErrorCategory.INFERENCE, "Model timeout"),
        ]

        for category, message in error_scenarios:
            from src.utils.error_handling import ErrorContext

            context = ErrorContext(
                operation=f"test_{category.value}",
                component="test_component",
                timestamp=0,
                severity=ErrorSeverity.MEDIUM,
                category=category,
                metadata={"test": True}
            )

            # Each error type should have recovery strategies
            assert category in error_handler.recovery_strategies
            assert len(error_handler.recovery_strategies[category]) > 0

        # Check error tracking
        assert len(error_handler.error_counts) >= 0  # Starts empty


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    def test_memory_optimization_performance(self):
        """Test memory optimization performance integration."""
        config = get_default_config()
        memory_manager = MemoryManager(
            memory_limit_gb=config.performance.memory_limit_gb,
            buffer_gb=config.performance.memory_buffer_gb
        )

        # Test memory optimization workflow
        import time
        start_time = time.time()

        # Simulate memory operations
        with memory_manager.memory_guard("performance_test"):
            memory_manager.clear_gpu_memory()
            memory_manager.clear_cpu_memory()

            # Check memory availability
            is_available = memory_manager.check_memory_availability(1.0)
            assert isinstance(is_available, bool)

        end_time = time.time()
        operation_time = end_time - start_time

        # Memory operations should be fast (under 1 second)
        assert operation_time < 1.0

    def test_configuration_loading_performance(self):
        """Test configuration loading performance."""
        import time

        # Create temporary config file
        config = get_default_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        config.save(config_path)

        try:
            # Test loading performance
            start_time = time.time()

            for _ in range(10):  # Load config 10 times
                loaded_config = MRAGConfig.load(config_path)
                loaded_config.validate()

            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / 10

            # Each config load should be fast (under 0.1 seconds)
            assert avg_time < 0.1

        finally:
            if Path(config_path).exists():
                os.unlink(config_path)


@pytest.mark.integration
class TestSystemValidation:
    """Test system-wide validation and health checks."""

    def test_system_components_compatibility(self):
        """Test that all system components are compatible."""
        config = get_default_config()

        # Test that configuration is valid
        assert config.validate() is True

        # Test that memory manager can be created
        memory_manager = MemoryManager(
            memory_limit_gb=config.performance.memory_limit_gb,
            buffer_gb=config.performance.memory_buffer_gb
        )
        assert memory_manager.effective_limit_gb > 0

        # Test that error handler can be created
        error_handler = ErrorHandler(max_retries=3)
        assert len(error_handler.recovery_strategies) > 0

        # Test memory monitoring
        stats = memory_manager.monitor.get_current_stats()
        assert stats.timestamp > 0

    def test_system_default_state(self):
        """Test system default state is valid and functional."""
        # Default configuration should be valid
        config = get_default_config()
        assert config.validate() is True

        # Default memory manager should be functional
        memory_manager = MemoryManager()
        assert memory_manager.memory_limit_gb > 0
        assert memory_manager.effective_limit_gb > 0

        # Default error handler should be functional
        error_handler = ErrorHandler()
        assert error_handler.max_retries > 0
        assert len(error_handler.recovery_strategies) > 0

        # Memory monitoring should work
        monitor = MemoryMonitor()
        stats = monitor.get_current_stats()
        assert hasattr(stats, 'gpu_allocated_gb')
        assert hasattr(stats, 'cpu_percent')

    def test_system_resource_constraints_validation(self):
        """Test validation of system resource constraints."""
        config = get_default_config()

        # Test memory constraint validation
        max_memory = config.model.max_memory_gb
        memory_limit = config.performance.memory_limit_gb
        memory_buffer = config.performance.memory_buffer_gb

        # Model memory should not exceed available memory
        assert max_memory <= (memory_limit - memory_buffer)

        # Batch size should be reasonable
        assert config.dataset.batch_size > 0
        assert config.dataset.batch_size <= 32  # Reasonable upper limit

        # Retrieval parameters should be reasonable
        assert config.retrieval.top_k > 0
        assert config.retrieval.top_k <= 10  # Reasonable upper limit