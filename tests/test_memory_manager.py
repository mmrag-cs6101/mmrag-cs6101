"""
Memory Management Tests

Tests for memory monitoring, optimization, and automatic cleanup functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.utils.memory_manager import MemoryManager, MemoryMonitor, MemoryStats


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_default_values(self):
        """Test default MemoryStats values."""
        stats = MemoryStats()
        assert stats.gpu_allocated_gb == 0.0
        assert stats.gpu_reserved_gb == 0.0
        assert stats.gpu_total_gb == 0.0
        assert stats.cpu_percent == 0.0
        assert stats.cpu_available_gb == 0.0
        assert stats.timestamp > 0

    def test_gpu_utilization_calculation(self):
        """Test GPU utilization percentage calculation."""
        stats = MemoryStats(
            gpu_allocated_gb=8.0,
            gpu_total_gb=16.0
        )
        assert stats.gpu_utilization_percent() == 50.0

        # Test with zero total memory
        stats_zero = MemoryStats(gpu_allocated_gb=4.0, gpu_total_gb=0.0)
        assert stats_zero.gpu_utilization_percent() == 0.0

    def test_memory_limit_check(self):
        """Test memory limit checking."""
        stats = MemoryStats(gpu_allocated_gb=12.0)

        assert stats.is_within_limit(15.0) is True
        assert stats.is_within_limit(10.0) is False


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_initialization(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor(memory_limit_gb=14.0, check_interval=2.0)
        assert monitor.memory_limit_gb == 14.0
        assert monitor.check_interval == 2.0
        assert monitor.monitoring_active is False
        assert len(monitor.stats_history) == 0

    @patch('src.utils.memory_manager.TORCH_AVAILABLE', True)
    @patch('src.utils.memory_manager.torch')
    @patch('src.utils.memory_manager.psutil')
    def test_get_current_stats_with_gpu(self, mock_psutil, mock_torch):
        """Test getting current stats with GPU available."""
        # Mock torch CUDA functions
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 8 * 1024**3  # 8GB
        mock_torch.cuda.memory_reserved.return_value = 10 * 1024**3  # 10GB
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16 * 1024**3  # 16GB

        # Mock psutil
        mock_memory = Mock()
        mock_memory.percent = 75.0
        mock_memory.available = 4 * 1024**3  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory

        monitor = MemoryMonitor()
        stats = monitor.get_current_stats()

        assert stats.gpu_allocated_gb == 8.0
        assert stats.gpu_reserved_gb == 10.0
        assert stats.gpu_total_gb == 16.0
        assert stats.cpu_percent == 75.0
        assert stats.cpu_available_gb == 4.0

    @patch('src.utils.memory_manager.TORCH_AVAILABLE', False)
    @patch('src.utils.memory_manager.psutil')
    def test_get_current_stats_without_gpu(self, mock_psutil):
        """Test getting current stats without GPU."""
        # Mock psutil
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory

        monitor = MemoryMonitor()
        stats = monitor.get_current_stats()

        assert stats.gpu_allocated_gb == 0.0
        assert stats.gpu_reserved_gb == 0.0
        assert stats.gpu_total_gb == 0.0
        assert stats.cpu_percent == 60.0
        assert stats.cpu_available_gb == 8.0

    def test_memory_pressure_check(self):
        """Test memory pressure detection."""
        monitor = MemoryMonitor(memory_limit_gb=10.0)

        with patch.object(monitor, 'get_current_stats') as mock_stats:
            # Test normal memory usage
            mock_stats.return_value = MemoryStats(gpu_allocated_gb=8.0)
            assert monitor.check_memory_pressure() is False

            # Test high memory usage
            mock_stats.return_value = MemoryStats(gpu_allocated_gb=12.0)
            assert monitor.check_memory_pressure() is True

    def test_memory_usage_trend(self):
        """Test memory usage trend analysis."""
        monitor = MemoryMonitor()

        # Add some sample stats
        for i in range(5):
            stats = MemoryStats(gpu_allocated_gb=float(i + 5))  # 5GB to 9GB
            monitor.stats_history.append(stats)

        trend = monitor.get_memory_usage_trend(window_size=5)

        assert trend["trend"] == 4.0  # 9GB - 5GB
        assert trend["avg_usage"] == 7.0  # (5+6+7+8+9)/5
        assert trend["peak_usage"] == 9.0
        assert trend["min_usage"] == 5.0

    def test_memory_usage_trend_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        monitor = MemoryMonitor()

        # No data
        trend = monitor.get_memory_usage_trend()
        assert trend["trend"] == 0.0
        assert trend["avg_usage"] == 0.0
        assert trend["peak_usage"] == 0.0

        # Single data point
        monitor.stats_history.append(MemoryStats(gpu_allocated_gb=5.0))
        trend = monitor.get_memory_usage_trend()
        assert trend["trend"] == 0.0
        assert trend["avg_usage"] == 5.0


class TestMemoryManager:
    """Test MemoryManager class."""

    def test_initialization(self):
        """Test MemoryManager initialization."""
        manager = MemoryManager(memory_limit_gb=14.0, buffer_gb=2.0)
        assert manager.memory_limit_gb == 14.0
        assert manager.buffer_gb == 2.0
        assert manager.effective_limit_gb == 12.0
        assert isinstance(manager.monitor, MemoryMonitor)

    @patch('src.utils.memory_manager.TORCH_AVAILABLE', True)
    @patch('src.utils.memory_manager.torch')
    def test_clear_gpu_memory(self, mock_torch):
        """Test GPU memory clearing."""
        mock_torch.cuda.is_available.return_value = True

        manager = MemoryManager()

        # Test normal cleanup
        manager.clear_gpu_memory(aggressive=False)
        mock_torch.cuda.empty_cache.assert_called()

        # Test aggressive cleanup
        manager.clear_gpu_memory(aggressive=True)
        mock_torch.cuda.synchronize.assert_called()
        assert mock_torch.cuda.empty_cache.call_count >= 2

    @patch('src.utils.memory_manager.TORCH_AVAILABLE', False)
    def test_clear_gpu_memory_no_torch(self):
        """Test GPU memory clearing without PyTorch."""
        manager = MemoryManager()
        # Should not raise any exceptions
        manager.clear_gpu_memory(aggressive=True)

    @patch('src.utils.memory_manager.gc')
    def test_clear_cpu_memory(self, mock_gc):
        """Test CPU memory clearing."""
        manager = MemoryManager()
        manager.clear_cpu_memory()
        mock_gc.collect.assert_called_once()

    @patch('src.utils.memory_manager.gc')
    def test_emergency_cleanup(self, mock_gc):
        """Test emergency memory cleanup."""
        manager = MemoryManager()

        with patch.object(manager, 'clear_gpu_memory') as mock_gpu_clear:
            manager.emergency_cleanup()

            mock_gpu_clear.assert_called_once_with(aggressive=True)
            assert mock_gc.collect.call_count == 3  # Called 3 times

    def test_check_memory_availability(self):
        """Test memory availability checking."""
        manager = MemoryManager(memory_limit_gb=16.0, buffer_gb=1.0)  # 15GB effective

        with patch.object(manager.monitor, 'get_current_stats') as mock_stats:
            # Test sufficient memory
            mock_stats.return_value = MemoryStats(gpu_allocated_gb=5.0)
            assert manager.check_memory_availability(8.0) is True  # 15-5=10 > 8

            # Test insufficient memory
            mock_stats.return_value = MemoryStats(gpu_allocated_gb=12.0)
            assert manager.check_memory_availability(5.0) is False  # 15-12=3 < 5

    def test_get_recommended_batch_size(self):
        """Test batch size recommendation based on memory."""
        manager = MemoryManager()

        # Test different memory scenarios
        assert manager.get_recommended_batch_size(8, 500.0) == 8  # Normal case
        assert manager.get_recommended_batch_size(8, 1000.0) == min(8, 1)  # High memory per item
        assert manager.get_recommended_batch_size(1, 100.0) == 1  # Minimum batch size

    def test_memory_guard_context_manager(self):
        """Test memory guard context manager."""
        manager = MemoryManager()

        with patch.object(manager.monitor, 'log_memory_stats') as mock_log:
            with patch.object(manager, 'clear_gpu_memory') as mock_clear:
                # Mock initial and final stats
                initial_stats = MemoryStats(gpu_allocated_gb=5.0)
                final_stats = MemoryStats(gpu_allocated_gb=5.2)
                mock_log.side_effect = [initial_stats, final_stats]

                with manager.memory_guard("test_operation"):
                    pass  # Normal execution

                assert mock_log.call_count == 2
                mock_clear.assert_called_once()

    def test_memory_guard_with_exception(self):
        """Test memory guard with exception handling."""
        manager = MemoryManager()

        with patch.object(manager, 'emergency_cleanup') as mock_emergency:
            with pytest.raises(ValueError):
                with manager.memory_guard("test_operation"):
                    raise ValueError("Test exception")

            mock_emergency.assert_called_once()

    def test_memory_guard_leak_detection(self):
        """Test memory leak detection in memory guard."""
        manager = MemoryManager()

        with patch.object(manager.monitor, 'log_memory_stats') as mock_log:
            with patch.object(manager, 'clear_gpu_memory'):
                # Mock stats showing memory leak (>500MB increase)
                initial_stats = MemoryStats(gpu_allocated_gb=5.0)
                final_stats = MemoryStats(gpu_allocated_gb=6.0)  # 1GB increase
                mock_log.side_effect = [initial_stats, final_stats]

                with manager.memory_guard("test_operation"):
                    pass

                # Should detect potential memory leak
                assert mock_log.call_count == 2


@pytest.mark.integration
class TestMemoryManagerIntegration:
    """Integration tests for memory management."""

    def test_memory_monitoring_workflow(self):
        """Test complete memory monitoring workflow."""
        manager = MemoryManager(memory_limit_gb=16.0)

        # Simulate memory usage pattern
        with patch.object(manager.monitor, 'get_current_stats') as mock_stats:
            # Start with low memory
            mock_stats.return_value = MemoryStats(gpu_allocated_gb=2.0)

            # Check availability
            assert manager.check_memory_availability(10.0) is True

            # Simulate memory increase
            mock_stats.return_value = MemoryStats(gpu_allocated_gb=14.0)

            # Check memory pressure
            assert manager.monitor.check_memory_pressure() is True

            # Perform cleanup
            manager.clear_gpu_memory()

            # Check recommended batch size with high memory usage
            recommended_size = manager.get_recommended_batch_size(8, 500.0)
            assert recommended_size <= 8

    @patch('src.utils.memory_manager.TORCH_AVAILABLE', True)
    @patch('src.utils.memory_manager.torch')
    def test_gpu_optimization_workflow(self, mock_torch):
        """Test GPU optimization workflow."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16 * 1024**3

        manager = MemoryManager()

        # Test optimization sequence
        manager.optimize_memory_allocation()
        manager.clear_gpu_memory(aggressive=False)

        # Verify torch functions were called
        mock_torch.cuda.is_available.assert_called()
        mock_torch.cuda.empty_cache.assert_called()