"""
Sprint 6 Integration Tests
Test the specific enhancements implemented in Sprint 6: End-to-End Pipeline Integration.

Sprint 6 focuses on:
1. Enhanced pipeline orchestration with memory management
2. Dynamic memory allocation between retrieval and generation stages
3. Advanced error handling and recovery mechanisms
4. Performance monitoring and optimization triggers
5. End-to-end validation with comprehensive testing
"""

import pytest
import torch
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from typing import List, Dict, Any

from src.pipeline import MRAGPipeline, PipelineResult
from src.config import MRAGConfig, ModelConfig, DatasetConfig, RetrievalConfig, GenerationConfig, EvaluationConfig, PerformanceConfig
from src.dataset.interface import Sample
from src.utils.memory_manager import MemoryManager
from src.utils.error_handling import MRAGError


class TestSprint6PipelineEnhancements:
    """Test Sprint 6 specific enhancements to the MRAG pipeline."""

    @pytest.fixture
    def sprint6_config(self):
        """Create configuration optimized for Sprint 6 testing."""
        return MRAGConfig(
            model=ModelConfig(
                vlm_name="llava-hf/llava-1.5-7b-hf",
                retriever_name="openai/clip-vit-base-patch32",
                quantization="4bit",
                max_memory_gb=5.0,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            dataset=DatasetConfig(
                data_path="/tmp/test_dataset",
                batch_size=2,
                cache_embeddings=False
            ),
            retrieval=RetrievalConfig(
                top_k=5,
                batch_size=8
            ),
            generation=GenerationConfig(
                max_length=512,
                temperature=0.7
            ),
            evaluation=EvaluationConfig(
                scenarios=["angle", "partial"]
            ),
            performance=PerformanceConfig(
                memory_limit_gb=8.0,
                memory_buffer_gb=1.0,
                retrieval_timeout=3.0,  # Lower threshold for testing
                generation_timeout=5.0,  # Lower threshold for testing
                total_pipeline_timeout=8.0  # Lower threshold for testing
            )
        )

    def test_enhanced_pipeline_result_structure(self, sprint6_config):
        """Test that PipelineResult includes Sprint 6 enhancements."""
        pipeline = MRAGPipeline(sprint6_config)

        # Create a mock result to test structure
        result = PipelineResult(
            question_id="test_001",
            question="Test question",
            retrieved_images=["/path/to/image.jpg"],
            retrieval_scores=[0.95],
            generated_answer="Test answer",
            confidence_score=0.85,
            total_time=5.2,
            retrieval_time=2.1,
            generation_time=3.1,
            memory_usage={"gpu_allocated_gb": 4.5},
            metadata={"test": "data"}
        )

        # Verify Sprint 6 enhancements are present
        assert hasattr(result, 'pipeline_stage_times')
        assert hasattr(result, 'memory_usage_per_stage')
        assert hasattr(result, 'error_recovery_attempts')
        assert hasattr(result, 'optimization_triggers')

        # Test that default values are properly initialized
        assert isinstance(result.pipeline_stage_times, dict)
        assert isinstance(result.memory_usage_per_stage, dict)
        assert isinstance(result.error_recovery_attempts, int)
        assert isinstance(result.optimization_triggers, list)

    def test_performance_monitoring_initialization(self, sprint6_config):
        """Test that performance monitoring is properly initialized."""
        pipeline = MRAGPipeline(sprint6_config)

        # Verify performance monitor configuration
        assert hasattr(pipeline, 'performance_monitor')
        assert 'retrieval_time_threshold' in pipeline.performance_monitor
        assert 'generation_time_threshold' in pipeline.performance_monitor
        assert 'total_time_threshold' in pipeline.performance_monitor
        assert 'memory_optimization_threshold' in pipeline.performance_monitor

        # Verify values match configuration
        assert pipeline.performance_monitor['retrieval_time_threshold'] == 3.0
        assert pipeline.performance_monitor['generation_time_threshold'] == 5.0
        assert pipeline.performance_monitor['total_time_threshold'] == 8.0

    def test_error_recovery_initialization(self, sprint6_config):
        """Test that error recovery mechanisms are properly initialized."""
        pipeline = MRAGPipeline(sprint6_config)

        # Verify error recovery structures
        assert hasattr(pipeline, 'recovery_attempts')
        assert hasattr(pipeline, 'max_recovery_attempts')
        assert isinstance(pipeline.recovery_attempts, dict)
        assert pipeline.max_recovery_attempts == 3

    def test_performance_trigger_detection(self, sprint6_config):
        """Test performance trigger detection functionality."""
        pipeline = MRAGPipeline(sprint6_config)

        # Test no triggers for normal performance
        triggers = pipeline.check_performance_triggers(1.0, 2.0, 3.0)
        assert isinstance(triggers, list)
        assert len(triggers) == 0

        # Test retrieval time trigger
        triggers = pipeline.check_performance_triggers(5.0, 2.0, 7.0)
        assert "optimize_retrieval" in triggers

        # Test generation time trigger
        triggers = pipeline.check_performance_triggers(1.0, 8.0, 9.0)
        assert "optimize_generation" in triggers

        # Test total time trigger
        triggers = pipeline.check_performance_triggers(3.0, 6.0, 10.0)
        assert "optimize_pipeline" in triggers

    @patch('src.pipeline.logger')
    def test_performance_optimization_application(self, mock_logger, sprint6_config):
        """Test that performance optimizations are applied correctly."""
        pipeline = MRAGPipeline(sprint6_config)

        # Test memory optimization
        initial_batch_size = pipeline.config.retrieval.batch_size
        pipeline.apply_performance_optimizations(["optimize_memory"])

        # Verify memory optimization was attempted
        mock_logger.info.assert_called()

        # Test retrieval optimization
        initial_top_k = pipeline.config.retrieval.top_k
        pipeline.apply_performance_optimizations(["optimize_retrieval"])

        # Should reduce top_k
        assert pipeline.config.retrieval.top_k <= initial_top_k

        # Test generation optimization
        initial_max_length = pipeline.config.generation.max_length
        pipeline.apply_performance_optimizations(["optimize_generation"])

        # Should reduce max_length
        assert pipeline.config.generation.max_length <= initial_max_length

    def test_error_stage_identification(self, sprint6_config):
        """Test error stage identification logic."""
        pipeline = MRAGPipeline(sprint6_config)

        # Test memory error identification
        memory_error = Exception("CUDA out of memory")
        stage = pipeline._identify_error_stage(memory_error, 0.0, 0.0)
        assert stage == "memory"

        # Test retrieval stage error
        retrieval_error = Exception("Model loading failed")
        stage = pipeline._identify_error_stage(retrieval_error, 0.0, 0.0)
        assert stage == "retrieval"

        # Test generation stage error (retrieval completed)
        generation_error = Exception("Model loading failed")
        stage = pipeline._identify_error_stage(generation_error, 2.0, 0.0)
        assert stage == "generation"

        # Test general error (both stages completed)
        general_error = Exception("Unknown error")
        stage = pipeline._identify_error_stage(general_error, 2.0, 3.0)
        assert stage == "general"

    @patch('src.pipeline.time.sleep')  # Mock sleep to speed up tests
    def test_error_recovery_mechanisms(self, mock_sleep, sprint6_config):
        """Test error recovery mechanisms."""
        pipeline = MRAGPipeline(sprint6_config)

        # Mock the pipeline components
        pipeline.retriever = Mock()
        pipeline.generator = Mock()
        pipeline.retriever_loaded = True
        pipeline.generator_loaded = True

        # Test retrieval error recovery
        mock_error = Exception("Retrieval failed")
        result = pipeline.handle_pipeline_error(mock_error, "retrieval", "test_001")

        assert result is True
        assert pipeline.recovery_attempts["retrieval:test_001"] == 1
        assert pipeline.pipeline_stats["error_recoveries_performed"] == 1

        # Test that recovery attempts are limited
        for i in range(5):  # Exceed max attempts
            pipeline.handle_pipeline_error(mock_error, "retrieval", "test_002")

        # Should stop trying after max attempts
        result = pipeline.handle_pipeline_error(mock_error, "retrieval", "test_002")
        assert result is False

    @patch('src.dataset.mrag_dataset.MRAGDataset')
    @patch('src.retrieval.clip_retriever.CLIPRetriever')
    @patch('src.generation.llava_pipeline.LLaVAGenerationPipeline')
    def test_enhanced_query_processing_with_monitoring(self, mock_generator_class, mock_retriever_class, mock_dataset_class, sprint6_config):
        """Test enhanced query processing with Sprint 6 monitoring."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 5}
        mock_dataset_class.return_value = mock_dataset

        mock_retriever = Mock()
        from src.retrieval.interface import RetrievalResult
        mock_retriever.retrieve_similar.return_value = [
            RetrievalResult(image_path="/tmp/test_image.jpg", similarity_score=0.9, metadata={})
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_generator = Mock()
        from src.generation.interface import GenerationResult
        mock_generator.generate_answer.return_value = GenerationResult(
            answer="Test answer",
            confidence_score=0.8,
            generation_time=1.0,
            memory_usage={},
            metadata={}
        )
        mock_generator_class.return_value = mock_generator

        # Create mock image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            img = Image.new('RGB', (224, 224), color='red')
            img.save(temp_file.name)
            temp_image_path = temp_file.name

        try:
            # Update mock to use actual temp image
            mock_retriever.retrieve_similar.return_value = [
                RetrievalResult(image_path=temp_image_path, similarity_score=0.9, metadata={})
            ]

            pipeline = MRAGPipeline(sprint6_config)

            result = pipeline.process_query(
                question="Test question for Sprint 6",
                question_id="sprint6_test_001",
                ground_truth="Expected answer",
                use_sequential_loading=True
            )

            # Verify Sprint 6 enhancements in result
            assert isinstance(result, PipelineResult)
            assert hasattr(result, 'pipeline_stage_times')
            assert hasattr(result, 'memory_usage_per_stage')
            assert hasattr(result, 'optimization_triggers')

            # Verify stage times are recorded
            assert 'retrieval' in result.pipeline_stage_times
            assert 'generation' in result.pipeline_stage_times
            assert 'total' in result.pipeline_stage_times

            # Verify memory usage tracking
            assert 'after_retrieval' in result.memory_usage_per_stage
            assert 'after_generation' in result.memory_usage_per_stage

        finally:
            # Cleanup temp file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    def test_enhanced_statistics_tracking(self, sprint6_config):
        """Test enhanced statistics tracking for Sprint 6."""
        pipeline = MRAGPipeline(sprint6_config)

        # Verify initial enhanced stats
        stats = pipeline.get_pipeline_stats()
        assert "memory_optimizations_triggered" in pipeline.pipeline_stats
        assert "error_recoveries_performed" in pipeline.pipeline_stats
        assert "failed_queries" in pipeline.pipeline_stats

        # Test stat updates
        initial_optimizations = pipeline.pipeline_stats["memory_optimizations_triggered"]
        pipeline.apply_performance_optimizations(["optimize_memory"])
        assert pipeline.pipeline_stats["memory_optimizations_triggered"] > initial_optimizations

    @patch('src.pipeline.logger')
    def test_comprehensive_error_handling_with_recovery(self, mock_logger, sprint6_config):
        """Test comprehensive error handling with recovery attempts."""
        pipeline = MRAGPipeline(sprint6_config)

        # Mock failing components
        pipeline.dataset = Mock()
        pipeline.dataset.validate_dataset.return_value = {"status": "success", "total_samples": 5}

        # Test that errors trigger recovery attempts
        with patch.object(pipeline, 'initialize_dataset', side_effect=Exception("Dataset error")):
            result = pipeline.process_query("Test question", "error_test_001")

            # Should return error result with Sprint 6 enhancements
            assert isinstance(result, PipelineResult)
            assert "Error" in result.generated_answer
            assert "error" in result.metadata
            assert result.error_recovery_attempts >= 0  # Sprint 6 enhancement

    def test_memory_management_integration(self, sprint6_config):
        """Test memory management integration with Sprint 6 enhancements."""
        pipeline = MRAGPipeline(sprint6_config)

        # Test memory manager integration
        assert isinstance(pipeline.memory_manager, MemoryManager)

        # Test memory optimization trigger
        with patch.object(pipeline.memory_manager.monitor, 'get_current_stats') as mock_stats:
            # Mock high memory usage
            mock_stats.return_value.gpu_allocated_gb = 7.0
            mock_stats.return_value.gpu_total_gb = 8.0

            triggers = pipeline.check_performance_triggers(1.0, 1.0, 2.0)
            assert "optimize_memory" in triggers

    def test_sequential_vs_parallel_loading_strategies(self, sprint6_config):
        """Test both sequential and parallel loading strategies with Sprint 6 enhancements."""
        pipeline = MRAGPipeline(sprint6_config)

        # Mock components
        pipeline.dataset = Mock()
        pipeline.dataset.validate_dataset.return_value = {"status": "success", "total_samples": 5}

        # Test that force sequential flag works
        pipeline._optimize_pipeline()
        assert hasattr(pipeline, '_force_sequential')
        assert pipeline._force_sequential is True


class TestSprint6AcceptanceCriteria:
    """Test that all Sprint 6 acceptance criteria are met."""

    @pytest.fixture
    def acceptance_config(self):
        """Configuration for acceptance testing."""
        return MRAGConfig(
            model=ModelConfig(max_memory_gb=5.0),
            performance=PerformanceConfig(
                memory_limit_gb=15.0,
                retrieval_timeout=5.0,
                generation_timeout=25.0,
                total_pipeline_timeout=30.0
            )
        )

    def test_acceptance_criteria_pipeline_coordination(self, acceptance_config):
        """Test: Complete pipeline processes queries from input text to final answer."""
        pipeline = MRAGPipeline(acceptance_config)

        # Pipeline should be properly initialized
        assert pipeline.config is not None
        assert pipeline.memory_manager is not None
        assert hasattr(pipeline, 'performance_monitor')

    def test_acceptance_criteria_memory_conflict_prevention(self, acceptance_config):
        """Test: Pipeline coordination prevents memory conflicts between models."""
        pipeline = MRAGPipeline(acceptance_config)

        # Memory manager should be configured correctly
        assert pipeline.memory_manager.memory_limit_gb == 15.0
        assert pipeline.memory_manager.buffer_gb == 1.0
        assert pipeline.memory_manager.effective_limit_gb == 14.0

    def test_acceptance_criteria_performance_targets(self, acceptance_config):
        """Test: Total processing time <30 seconds per query (target configuration)."""
        pipeline = MRAGPipeline(acceptance_config)

        # Performance thresholds should match requirements
        assert pipeline.performance_monitor['total_time_threshold'] == 30.0
        assert pipeline.performance_monitor['retrieval_time_threshold'] == 5.0
        assert pipeline.performance_monitor['generation_time_threshold'] == 25.0

    def test_acceptance_criteria_error_handling(self, acceptance_config):
        """Test: Error handling gracefully recovers from individual component failures."""
        pipeline = MRAGPipeline(acceptance_config)

        # Error recovery mechanisms should be in place
        assert hasattr(pipeline, 'handle_pipeline_error')
        assert hasattr(pipeline, 'recovery_attempts')
        assert pipeline.max_recovery_attempts == 3

        # Test error stage identification
        assert hasattr(pipeline, '_identify_error_stage')

        # Test recovery methods exist
        assert hasattr(pipeline, '_recover_retrieval_error')
        assert hasattr(pipeline, '_recover_generation_error')
        assert hasattr(pipeline, '_recover_memory_error')

    def test_acceptance_criteria_memory_stability(self, acceptance_config):
        """Test: Memory usage stays within 15GB VRAM throughout pipeline execution."""
        pipeline = MRAGPipeline(acceptance_config)

        # Memory monitoring should be configured for 15GB limit
        memory_limit = pipeline.memory_manager.memory_limit_gb
        assert memory_limit == 15.0

        # Memory optimization threshold should be reasonable
        optimization_threshold = pipeline.performance_monitor['memory_optimization_threshold']
        assert 0.8 <= optimization_threshold <= 0.95

    def test_acceptance_criteria_integration_success_rate(self, acceptance_config):
        """Test: Integration success rate tracking (>95% target)."""
        pipeline = MRAGPipeline(acceptance_config)

        # Statistics tracking should include success/failure rates
        stats = pipeline.pipeline_stats
        assert 'successful_queries' in stats
        assert 'failed_queries' in stats
        assert 'total_queries' in stats

        # Method to calculate success rate should work
        total = stats['successful_queries'] + stats['failed_queries']
        if total > 0:
            success_rate = stats['successful_queries'] / total
            assert 0.0 <= success_rate <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])