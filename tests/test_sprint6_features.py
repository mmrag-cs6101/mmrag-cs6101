"""
Unit Tests for Sprint 6 Specific Features

Tests individual components and methods introduced in Sprint 6:
- Performance monitoring and optimization triggers
- Enhanced error handling and recovery
- Memory management integration
- Dynamic pipeline configuration
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import torch

from src.pipeline import MRAGPipeline, PipelineResult
from src.config import MRAGConfig, PerformanceConfig, ModelConfig, RetrievalConfig, GenerationConfig
from src.utils.memory_manager import MemoryManager, MemoryStats


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""

    @pytest.fixture
    def test_pipeline(self):
        """Create pipeline for testing."""
        config = MRAGConfig(
            performance=PerformanceConfig(
                retrieval_timeout=2.0,
                generation_timeout=5.0,
                total_pipeline_timeout=7.0
            )
        )
        return MRAGPipeline(config)

    def test_performance_monitor_initialization(self, test_pipeline):
        """Test performance monitor is properly initialized."""
        monitor = test_pipeline.performance_monitor

        assert monitor['retrieval_time_threshold'] == 2.0
        assert monitor['generation_time_threshold'] == 5.0
        assert monitor['total_time_threshold'] == 7.0
        assert monitor['memory_optimization_threshold'] == 0.9

    def test_no_triggers_for_normal_performance(self, test_pipeline):
        """Test no triggers are generated for normal performance."""
        triggers = test_pipeline.check_performance_triggers(1.0, 3.0, 4.0)
        assert triggers == []

    def test_retrieval_time_trigger(self, test_pipeline):
        """Test retrieval time threshold trigger."""
        triggers = test_pipeline.check_performance_triggers(3.0, 1.0, 4.0)
        assert "optimize_retrieval" in triggers

    def test_generation_time_trigger(self, test_pipeline):
        """Test generation time threshold trigger."""
        triggers = test_pipeline.check_performance_triggers(1.0, 6.0, 7.0)
        assert "optimize_generation" in triggers

    def test_total_time_trigger(self, test_pipeline):
        """Test total time threshold trigger."""
        triggers = test_pipeline.check_performance_triggers(3.0, 5.0, 8.0)
        assert "optimize_pipeline" in triggers

    def test_multiple_triggers(self, test_pipeline):
        """Test multiple triggers can be detected simultaneously."""
        triggers = test_pipeline.check_performance_triggers(3.0, 6.0, 9.0)
        assert "optimize_retrieval" in triggers
        assert "optimize_generation" in triggers
        assert "optimize_pipeline" in triggers

    @patch('src.pipeline.logger')
    def test_memory_pressure_trigger(self, mock_logger, test_pipeline):
        """Test memory pressure trigger detection."""
        # Mock high memory usage
        mock_stats = Mock()
        mock_stats.gpu_allocated_gb = 9.0
        mock_stats.gpu_total_gb = 10.0

        with patch.object(test_pipeline.memory_manager.monitor, 'get_current_stats', return_value=mock_stats):
            triggers = test_pipeline.check_performance_triggers(1.0, 1.0, 2.0)
            assert "optimize_memory" in triggers


class TestPerformanceOptimizations:
    """Test performance optimization implementations."""

    @pytest.fixture
    def test_pipeline(self):
        """Create pipeline with configurable parameters for testing."""
        config = MRAGConfig(
            retrieval=RetrievalConfig(top_k=10, batch_size=16),
            generation=GenerationConfig(max_length=1024)
        )
        return MRAGPipeline(config)

    def test_memory_optimization(self, test_pipeline):
        """Test memory optimization functionality."""
        # Mock loaded components
        test_pipeline.retriever_loaded = True
        test_pipeline.generator_loaded = True
        test_pipeline.retriever = Mock()
        test_pipeline.generator = Mock()

        initial_batch_size = test_pipeline.config.retrieval.batch_size

        test_pipeline._optimize_memory()

        # Should reduce batch size
        assert test_pipeline.config.retrieval.batch_size <= initial_batch_size

    def test_retrieval_optimization(self, test_pipeline):
        """Test retrieval optimization functionality."""
        initial_top_k = test_pipeline.config.retrieval.top_k

        test_pipeline._optimize_retrieval()

        # Should reduce top_k
        assert test_pipeline.config.retrieval.top_k < initial_top_k
        assert test_pipeline.config.retrieval.top_k >= 3  # Should not go below minimum

    def test_generation_optimization(self, test_pipeline):
        """Test generation optimization functionality."""
        initial_max_length = test_pipeline.config.generation.max_length

        test_pipeline._optimize_generation()

        # Should reduce max_length
        assert test_pipeline.config.generation.max_length < initial_max_length
        assert test_pipeline.config.generation.max_length >= 256  # Should not go below minimum

    def test_pipeline_optimization(self, test_pipeline):
        """Test global pipeline optimization."""
        test_pipeline._optimize_pipeline()

        # Should set force sequential flag
        assert hasattr(test_pipeline, '_force_sequential')
        assert test_pipeline._force_sequential is True

    @patch('src.pipeline.logger')
    def test_optimization_application(self, mock_logger, test_pipeline):
        """Test optimization application with logging."""
        test_pipeline.apply_performance_optimizations(["optimize_memory", "optimize_retrieval"])

        # Should increment optimization counter
        assert test_pipeline.pipeline_stats["memory_optimizations_triggered"] == 2

        # Should log optimization attempts
        mock_logger.info.assert_called()

    def test_optimization_error_handling(self, test_pipeline):
        """Test error handling in optimization application."""
        # Mock optimization method to raise error
        with patch.object(test_pipeline, '_optimize_memory', side_effect=Exception("Optimization failed")):
            # Should not crash
            test_pipeline.apply_performance_optimizations(["optimize_memory"])

            # Should still track attempt even if it failed
            assert test_pipeline.pipeline_stats["memory_optimizations_triggered"] == 1


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    @pytest.fixture
    def test_pipeline(self):
        """Create pipeline for error testing."""
        config = MRAGConfig()
        return MRAGPipeline(config)

    def test_error_stage_identification_memory_error(self, test_pipeline):
        """Test identification of memory errors."""
        error = Exception("CUDA out of memory")
        stage = test_pipeline._identify_error_stage(error, 0.0, 0.0)
        assert stage == "memory"

        error = Exception("out of memory: tried to allocate")
        stage = test_pipeline._identify_error_stage(error, 1.0, 1.0)
        assert stage == "memory"

    def test_error_stage_identification_model_loading(self, test_pipeline):
        """Test identification of model loading errors."""
        # Retrieval stage (no retrieval time recorded)
        error = Exception("Failed to load model tokenizer")
        stage = test_pipeline._identify_error_stage(error, 0.0, 0.0)
        assert stage == "retrieval"

        # Generation stage (retrieval completed)
        stage = test_pipeline._identify_error_stage(error, 2.0, 0.0)
        assert stage == "generation"

    def test_error_stage_identification_by_timing(self, test_pipeline):
        """Test error stage identification based on timing."""
        error = Exception("Unknown error")

        # Error during retrieval
        stage = test_pipeline._identify_error_stage(error, 0.0, 0.0)
        assert stage == "retrieval"

        # Error during generation
        stage = test_pipeline._identify_error_stage(error, 2.0, 0.0)
        assert stage == "generation"

        # Error after both stages
        stage = test_pipeline._identify_error_stage(error, 2.0, 3.0)
        assert stage == "general"

    @patch('src.pipeline.time.sleep')  # Mock sleep to speed up tests
    def test_error_recovery_tracking(self, mock_sleep, test_pipeline):
        """Test error recovery attempt tracking."""
        error = Exception("Test error")

        # First recovery attempt
        result = test_pipeline.handle_pipeline_error(error, "retrieval", "test_001")
        assert result is True
        assert test_pipeline.recovery_attempts["retrieval:test_001"] == 1

        # Second attempt
        test_pipeline.handle_pipeline_error(error, "retrieval", "test_001")
        assert test_pipeline.recovery_attempts["retrieval:test_001"] == 2

        # Should track error recovery stats
        assert test_pipeline.pipeline_stats["error_recoveries_performed"] == 2

    @patch('src.pipeline.time.sleep')
    def test_recovery_attempt_limit(self, mock_sleep, test_pipeline):
        """Test recovery attempt limits."""
        error = Exception("Persistent error")

        # Exhaust recovery attempts
        for i in range(test_pipeline.max_recovery_attempts):
            result = test_pipeline.handle_pipeline_error(error, "test_stage", "test_002")
            assert result is True

        # Next attempt should fail
        result = test_pipeline.handle_pipeline_error(error, "test_stage", "test_002")
        assert result is False

    @patch('src.pipeline.time.sleep')
    def test_retrieval_error_recovery(self, mock_sleep, test_pipeline):
        """Test retrieval-specific error recovery."""
        test_pipeline.retriever = Mock()
        test_pipeline.retriever_loaded = True

        error = Exception("Retrieval error")
        result = test_pipeline._recover_retrieval_error(error)

        assert result is True
        # Should attempt to reload retriever (this would call methods on mocked retriever)

    @patch('src.pipeline.time.sleep')
    def test_generation_error_recovery(self, mock_sleep, test_pipeline):
        """Test generation-specific error recovery."""
        test_pipeline.generator = Mock()
        test_pipeline.generator_loaded = True

        error = Exception("Generation error")
        result = test_pipeline._recover_generation_error(error)

        assert result is True
        # Should attempt to reload generator

    def test_memory_error_recovery(self, test_pipeline):
        """Test memory-specific error recovery."""
        test_pipeline.retriever = Mock()
        test_pipeline.generator = Mock()
        test_pipeline.retriever_loaded = True
        test_pipeline.generator_loaded = True

        error = Exception("Memory error")
        result = test_pipeline._recover_memory_error(error)

        assert result is True
        # Should unload both models
        assert test_pipeline.retriever_loaded is False
        assert test_pipeline.generator_loaded is False

    def test_general_error_recovery(self, test_pipeline):
        """Test general error recovery."""
        error = Exception("General error")
        result = test_pipeline._recover_general_error(error)

        assert result is True  # General recovery should always return True


class TestEnhancedStatistics:
    """Test enhanced statistics and monitoring."""

    @pytest.fixture
    def test_pipeline(self):
        """Create pipeline for statistics testing."""
        return MRAGPipeline(MRAGConfig())

    def test_enhanced_statistics_initialization(self, test_pipeline):
        """Test that enhanced statistics are properly initialized."""
        stats = test_pipeline.pipeline_stats

        # Original stats
        assert "total_queries" in stats
        assert "successful_queries" in stats
        assert "avg_retrieval_time" in stats
        assert "avg_generation_time" in stats
        assert "avg_total_time" in stats

        # Sprint 6 enhancements
        assert "failed_queries" in stats
        assert "memory_optimizations_triggered" in stats
        assert "error_recoveries_performed" in stats

        # Should start at zero
        assert stats["failed_queries"] == 0
        assert stats["memory_optimizations_triggered"] == 0
        assert stats["error_recoveries_performed"] == 0

    def test_statistics_update_on_optimization(self, test_pipeline):
        """Test statistics update when optimizations are triggered."""
        initial_count = test_pipeline.pipeline_stats["memory_optimizations_triggered"]

        test_pipeline.apply_performance_optimizations(["optimize_memory"])

        assert test_pipeline.pipeline_stats["memory_optimizations_triggered"] == initial_count + 1

    def test_statistics_update_on_error_recovery(self, test_pipeline):
        """Test statistics update when error recovery is performed."""
        initial_count = test_pipeline.pipeline_stats["error_recoveries_performed"]

        with patch('src.pipeline.time.sleep'):
            test_pipeline.handle_pipeline_error(Exception("Test"), "test_stage", "test_id")

        assert test_pipeline.pipeline_stats["error_recoveries_performed"] == initial_count + 1

    def test_get_pipeline_stats_includes_memory_info(self, test_pipeline):
        """Test that pipeline stats include memory information."""
        stats = test_pipeline.get_pipeline_stats()

        assert "memory_stats" in stats
        assert "memory_trend" in stats

        # Memory stats should be a dictionary
        assert isinstance(stats["memory_stats"], dict)


class TestEnhancedPipelineResult:
    """Test enhanced PipelineResult structure."""

    def test_pipeline_result_default_initialization(self):
        """Test PipelineResult initializes Sprint 6 fields correctly."""
        result = PipelineResult(
            question_id="test",
            question="test question",
            retrieved_images=[],
            retrieval_scores=[],
            generated_answer="test answer",
            confidence_score=0.5,
            total_time=1.0,
            retrieval_time=0.5,
            generation_time=0.5,
            memory_usage={},
            metadata={}
        )

        # Sprint 6 fields should be initialized
        assert isinstance(result.pipeline_stage_times, dict)
        assert isinstance(result.memory_usage_per_stage, dict)
        assert isinstance(result.error_recovery_attempts, int)
        assert isinstance(result.optimization_triggers, list)

        # Should be empty/zero by default
        assert len(result.pipeline_stage_times) == 0
        assert len(result.memory_usage_per_stage) == 0
        assert result.error_recovery_attempts == 0
        assert len(result.optimization_triggers) == 0

    def test_pipeline_result_with_sprint6_data(self):
        """Test PipelineResult with Sprint 6 data populated."""
        stage_times = {"retrieval": 1.0, "generation": 2.0}
        memory_per_stage = {"after_retrieval": {"gpu_gb": 3.0}}
        triggers = ["optimize_memory"]

        result = PipelineResult(
            question_id="test",
            question="test question",
            retrieved_images=[],
            retrieval_scores=[],
            generated_answer="test answer",
            confidence_score=0.5,
            total_time=3.0,
            retrieval_time=1.0,
            generation_time=2.0,
            memory_usage={},
            metadata={},
            pipeline_stage_times=stage_times,
            memory_usage_per_stage=memory_per_stage,
            error_recovery_attempts=1,
            optimization_triggers=triggers
        )

        assert result.pipeline_stage_times == stage_times
        assert result.memory_usage_per_stage == memory_per_stage
        assert result.error_recovery_attempts == 1
        assert result.optimization_triggers == triggers


class TestMemoryManagementIntegration:
    """Test memory management integration with Sprint 6 features."""

    @pytest.fixture
    def test_pipeline(self):
        """Create pipeline with memory management for testing."""
        config = MRAGConfig(
            performance=PerformanceConfig(
                memory_limit_gb=8.0,
                memory_buffer_gb=1.0
            )
        )
        return MRAGPipeline(config)

    def test_memory_manager_initialization(self, test_pipeline):
        """Test memory manager is properly initialized."""
        memory_manager = test_pipeline.memory_manager

        assert isinstance(memory_manager, MemoryManager)
        assert memory_manager.memory_limit_gb == 8.0
        assert memory_manager.buffer_gb == 1.0
        assert memory_manager.effective_limit_gb == 7.0

    def test_memory_optimization_integration(self, test_pipeline):
        """Test memory optimization integrates with memory manager."""
        # Mock high memory usage scenario
        with patch.object(test_pipeline.memory_manager, 'clear_gpu_memory') as mock_clear:
            test_pipeline._optimize_memory()

            # Should call memory clearing
            mock_clear.assert_called_with(aggressive=True)

    def test_memory_recovery_integration(self, test_pipeline):
        """Test memory error recovery integrates with memory manager."""
        with patch.object(test_pipeline.memory_manager, 'emergency_cleanup') as mock_cleanup:
            test_pipeline._recover_memory_error(Exception("Memory error"))

            # Should call emergency cleanup
            mock_cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])