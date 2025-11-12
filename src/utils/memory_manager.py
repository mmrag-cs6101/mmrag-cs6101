"""
Memory Management Utilities

Provides VRAM monitoring, memory optimization, and automatic cleanup for MRAG-Bench system.
"""

import gc
import time
import psutil
import logging
from typing import Dict, Optional, Any, List
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_total_gb: float = 0.0
    cpu_percent: float = 0.0
    cpu_available_gb: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def gpu_utilization_percent(self) -> float:
        """Calculate GPU memory utilization percentage."""
        if self.gpu_total_gb > 0:
            return (self.gpu_allocated_gb / self.gpu_total_gb) * 100
        return 0.0

    def is_within_limit(self, limit_gb: float) -> bool:
        """Check if GPU memory is within specified limit."""
        return self.gpu_allocated_gb <= limit_gb


class MemoryMonitor:
    """Real-time memory monitoring and alerting."""

    def __init__(self, memory_limit_gb: float = 15.0, check_interval: float = 1.0):
        """
        Initialize memory monitor.

        Args:
            memory_limit_gb: GPU memory limit in GB
            check_interval: Monitoring check interval in seconds
        """
        self.memory_limit_gb = memory_limit_gb
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.stats_history: List[MemoryStats] = []

    def get_current_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        stats = MemoryStats()

        # GPU memory stats
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats.gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
            stats.gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
            stats.gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # CPU memory stats
        memory_info = psutil.virtual_memory()
        stats.cpu_percent = memory_info.percent
        stats.cpu_available_gb = memory_info.available / 1024**3

        return stats

    def log_memory_stats(self, prefix: str = "") -> MemoryStats:
        """Log current memory statistics."""
        stats = self.get_current_stats()
        self.stats_history.append(stats)

        if prefix:
            prefix = f"{prefix}: "

        self.logger.info(
            f"{prefix}GPU Memory - Allocated: {stats.gpu_allocated_gb:.2f}GB, "
            f"Reserved: {stats.gpu_reserved_gb:.2f}GB, "
            f"Total: {stats.gpu_total_gb:.2f}GB "
            f"({stats.gpu_utilization_percent():.1f}% utilization)"
        )

        return stats

    def check_memory_pressure(self) -> bool:
        """
        Check if system is under memory pressure.

        Returns:
            True if memory usage is critical, False otherwise
        """
        stats = self.get_current_stats()
        return not stats.is_within_limit(self.memory_limit_gb)

    def get_memory_usage_trend(self, window_size: int = 10) -> Dict[str, float]:
        """
        Analyze memory usage trend over recent measurements.

        Args:
            window_size: Number of recent measurements to analyze

        Returns:
            Dictionary with trend analysis
        """
        if len(self.stats_history) < 2:
            return {"trend": 0.0, "avg_usage": 0.0, "peak_usage": 0.0}

        recent_stats = self.stats_history[-window_size:]
        gpu_usage = [s.gpu_allocated_gb for s in recent_stats]

        return {
            "trend": gpu_usage[-1] - gpu_usage[0] if len(gpu_usage) > 1 else 0.0,
            "avg_usage": sum(gpu_usage) / len(gpu_usage),
            "peak_usage": max(gpu_usage),
            "min_usage": min(gpu_usage)
        }


class MemoryManager:
    """Automatic memory management and optimization."""

    def __init__(self, memory_limit_gb: float = 15.0, buffer_gb: float = 1.0):
        """
        Initialize memory manager.

        Args:
            memory_limit_gb: Maximum GPU memory limit
            buffer_gb: Safety buffer for memory allocation
        """
        self.memory_limit_gb = memory_limit_gb
        self.buffer_gb = buffer_gb
        self.effective_limit_gb = memory_limit_gb - buffer_gb
        self.monitor = MemoryMonitor(self.effective_limit_gb)
        self.logger = logging.getLogger(__name__)

    def clear_gpu_memory(self, aggressive: bool = False) -> None:
        """
        Clear GPU memory cache.

        Args:
            aggressive: If True, perform aggressive cleanup
        """
        if not TORCH_AVAILABLE:
            return

        if torch.cuda.is_available():
            # Standard cleanup
            torch.cuda.empty_cache()

            if aggressive:
                # Aggressive cleanup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()

            self.logger.debug("GPU memory cache cleared")

    def clear_cpu_memory(self) -> None:
        """Clear CPU memory by forcing garbage collection."""
        gc.collect()
        self.logger.debug("CPU garbage collection performed")

    def emergency_cleanup(self) -> None:
        """Perform emergency memory cleanup."""
        self.logger.warning("Performing emergency memory cleanup")

        # Aggressive GPU cleanup
        self.clear_gpu_memory(aggressive=True)

        # Clear CPU memory
        self.clear_cpu_memory()

        # Force Python garbage collection multiple times
        for _ in range(3):
            gc.collect()

    def optimize_memory_allocation(self) -> None:
        """Optimize memory allocation settings."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return

        # Set memory fraction to prevent over-allocation
        memory_fraction = min(0.95, self.effective_limit_gb / self.monitor.get_current_stats().gpu_total_gb)

        # Note: PyTorch doesn't have a direct equivalent to TensorFlow's memory_limit
        # Instead, we'll use environment variables and careful memory management
        self.logger.info(f"Configured for {memory_fraction:.2f} GPU memory utilization")

    @contextmanager
    def memory_guard(self, operation_name: str = "operation"):
        """
        Context manager for automatic memory management.

        Args:
            operation_name: Name of the operation for logging
        """
        # Log initial memory state
        initial_stats = self.monitor.log_memory_stats(f"Before {operation_name}")

        try:
            yield
        except Exception as e:
            self.logger.error(f"Exception in {operation_name}: {e}")
            self.emergency_cleanup()
            raise
        finally:
            # Cleanup after operation
            self.clear_gpu_memory()
            final_stats = self.monitor.log_memory_stats(f"After {operation_name}")

            # Check for memory leaks
            memory_increase = final_stats.gpu_allocated_gb - initial_stats.gpu_allocated_gb
            if memory_increase > 2:  # 500MB threshold
                self.logger.warning(
                    f"Potential memory leak in {operation_name}: "
                    f"{memory_increase:.2f}GB increase"
                )

    def check_memory_availability(self, required_gb: float) -> bool:
        """
        Check if required memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if memory is available, False otherwise
        """
        current_stats = self.monitor.get_current_stats()
        available_gb = self.effective_limit_gb - current_stats.gpu_allocated_gb

        is_available = available_gb >= required_gb

        if not is_available:
            self.logger.warning(
                f"Insufficient memory: need {required_gb:.2f}GB, "
                f"available {available_gb:.2f}GB"
            )

        return is_available

    def get_recommended_batch_size(self, base_batch_size: int, memory_per_item_mb: float) -> int:
        """
        Get recommended batch size based on available memory.

        Args:
            base_batch_size: Desired batch size
            memory_per_item_mb: Memory requirement per batch item in MB

        Returns:
            Recommended batch size
        """
        current_stats = self.monitor.get_current_stats()
        available_gb = self.effective_limit_gb - current_stats.gpu_allocated_gb
        available_mb = available_gb * 1024

        max_batch_size = int(available_mb / memory_per_item_mb)
        recommended_size = min(base_batch_size, max_batch_size)

        if recommended_size < base_batch_size:
            self.logger.warning(
                f"Reducing batch size from {base_batch_size} to {recommended_size} "
                f"due to memory constraints"
            )

        return max(1, recommended_size)  # Ensure at least batch size of 1