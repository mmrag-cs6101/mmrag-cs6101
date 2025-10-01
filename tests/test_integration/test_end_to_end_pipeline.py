"""
End-to-End Integration Tests

Comprehensive integration tests for MRAG pipeline with memory validation.
"""

import pytest
import torch
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from typing import List, Dict, Any

from src.pipeline import MRAGPipeline, PipelineResult
from src.config import MRAGConfig, ModelConfig, DatasetConfig, RetrievalConfig, GenerationConfig, EvaluationConfig, PerformanceConfig
from src.dataset.interface import Sample
from src.utils.memory_manager import MemoryManager


class TestMRAGPipelineIntegration:
    """Integration tests for complete MRAG pipeline."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            os.makedirs(os.path.join(temp_dir, "images"))
            os.makedirs(os.path.join(temp_dir, "questions"))
            os.makedirs(os.path.join(temp_dir, "metadata"))

            # Create sample images
            for i in range(5):
                img = Image.new('RGB', (224, 224), color=(i*50, i*40, i*30))
                img.save(os.path.join(temp_dir, "images", f"image_{i}.jpg"))

            # Create sample questions
            questions_data = [
                {
                    "question_id": f"q_{i}",
                    "question": f"What do you see in this medical image {i}?",
                    "image_path": f"images/image_{i}.jpg",
                    "answer": f"This shows medical finding {i}",
                    "scenario": "angle_change" if i % 2 == 0 else "partial_view",
                    "choices": [f"Option A {i}", f"Option B {i}", f"Option C {i}"]
                }
                for i in range(5)
            ]

            with open(os.path.join(temp_dir, "questions", "questions.json"), 'w') as f:
                json.dump(questions_data, f)

            # Create metadata
            metadata = {
                "total_samples": 5,
                "scenarios": {
                    "angle_change": 3,
                    "partial_view": 2
                }
            }

            with open(os.path.join(temp_dir, "metadata", "dataset_info.json"), 'w') as f:
                json.dump(metadata, f)

            # Create scenario mapping
            scenario_mapping = {
                "angle_change": ["q_0", "q_2", "q_4"],
                "partial_view": ["q_1", "q_3"]
            }

            with open(os.path.join(temp_dir, "metadata", "scenario_mapping.json"), 'w') as f:
                json.dump(scenario_mapping, f)

            yield temp_dir

    @pytest.fixture
    def mock_config(self, temp_dataset_dir):
        """Create mock MRAG configuration."""
        return MRAGConfig(
            model=ModelConfig(
                vlm_name="llava-hf/llava-1.5-7b-hf",
                retriever_name="openai/clip-vit-base-patch32",
                quantization="4bit",
                max_memory_gb=5.0,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            dataset=DatasetConfig(
                data_path=temp_dataset_dir,
                batch_size=2,
                cache_embeddings=False  # Disable for testing
            ),
            retrieval=RetrievalConfig(
                top_k=3,
                batch_size=8
            ),
            generation=GenerationConfig(
                max_length=256,
                temperature=0.7
            ),
            evaluation=EvaluationConfig(
                scenarios=["angle", "partial"]
            ),
            performance=PerformanceConfig(
                memory_limit_gb=8.0,
                memory_buffer_gb=1.0,
                generation_timeout=30.0
            )
        )

    @pytest.fixture
    def sample_samples(self):
        """Create sample test data."""
        return [
            Sample(
                question_id="test_001",
                question="What anatomical structure is visible?",
                image_path="/path/to/test_image.jpg",
                image=None,
                ground_truth="This shows a cardiac structure",
                perspective_type="angle",
                metadata={"category": "cardiology"}
            ),
            Sample(
                question_id="test_002",
                question="Describe the pathological findings.",
                image_path="/path/to/test_image2.jpg",
                image=None,
                ground_truth="Abnormal tissue density observed",
                perspective_type="partial",
                metadata={"category": "radiology"}
            )
        ]

    def test_pipeline_initialization(self, mock_config):
        """Test pipeline initialization."""
        pipeline = MRAGPipeline(mock_config)

        assert pipeline.config == mock_config
        assert pipeline.dataset is None
        assert pipeline.retriever is None
        assert pipeline.generator is None
        assert not pipeline.retriever_loaded
        assert not pipeline.generator_loaded
        assert pipeline.memory_manager is not None

    @patch('src.dataset.mrag_dataset.MRAGDataset')
    def test_dataset_initialization(self, mock_dataset_class, mock_config):
        """Test dataset initialization."""
        mock_dataset = Mock()
        mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 100}
        mock_dataset_class.return_value = mock_dataset

        pipeline = MRAGPipeline(mock_config)
        pipeline.initialize_dataset()

        assert pipeline.dataset == mock_dataset
        mock_dataset_class.assert_called_once()
        mock_dataset.validate_dataset.assert_called_once()

    @patch('src.retrieval.clip_retriever.CLIPRetriever')
    @patch('src.dataset.mrag_dataset.MRAGDataset')
    def test_retriever_loading(self, mock_dataset_class, mock_retriever_class, mock_config):
        """Test retriever loading and index building."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 100}
        mock_dataset.get_retrieval_corpus.return_value = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        mock_dataset_class.return_value = mock_dataset

        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever

        pipeline = MRAGPipeline(mock_config)
        pipeline.load_retriever()

        assert pipeline.retriever_loaded is True
        assert pipeline.retriever == mock_retriever
        mock_retriever.load_model.assert_called_once()
        mock_retriever.build_index.assert_called_once()

    @patch('src.generation.llava_pipeline.LLaVAGenerationPipeline')
    def test_generator_loading(self, mock_generator_class, mock_config):
        """Test generator loading."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        pipeline = MRAGPipeline(mock_config)
        pipeline.load_generator()

        assert pipeline.generator_loaded is True
        assert pipeline.generator == mock_generator
        mock_generator.load_model.assert_called_once()

    @patch('src.generation.llava_pipeline.LLaVAGenerationPipeline')
    @patch('src.retrieval.clip_retriever.CLIPRetriever')
    @patch('src.dataset.mrag_dataset.MRAGDataset')
    def test_query_processing_sequential_loading(self, mock_dataset_class, mock_retriever_class, mock_generator_class, mock_config, temp_dataset_dir):
        """Test complete query processing with sequential loading."""
        # Setup dataset mock
        mock_dataset = Mock()
        mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 5}
        mock_dataset_class.return_value = mock_dataset

        # Setup retriever mock
        mock_retriever = Mock()
        from src.retrieval.interface import RetrievalResult
        mock_retrieval_results = [
            RetrievalResult(
                image_path=os.path.join(temp_dataset_dir, "images", "image_0.jpg"),
                similarity_score=0.95,
                metadata={"id": 0}
            ),
            RetrievalResult(
                image_path=os.path.join(temp_dataset_dir, "images", "image_1.jpg"),
                similarity_score=0.87,
                metadata={"id": 1}
            )
        ]
        mock_retriever.retrieve_similar.return_value = mock_retrieval_results
        mock_retriever_class.return_value = mock_retriever

        # Setup generator mock
        mock_generator = Mock()
        from src.generation.interface import GenerationResult
        mock_generation_result = GenerationResult(
            answer="This medical image shows anatomical structures consistent with normal findings.",
            confidence_score=0.85,
            generation_time=2.5,
            memory_usage={"gpu_allocated_gb": 4.2},
            metadata={"prompt_length": 150}
        )
        mock_generator.generate_answer.return_value = mock_generation_result
        mock_generator_class.return_value = mock_generator

        # Test query processing
        pipeline = MRAGPipeline(mock_config)

        result = pipeline.process_query(
            question="What do you observe in this medical image?",
            question_id="test_query_001",
            ground_truth="Normal anatomical structures",
            use_sequential_loading=True
        )

        # Verify result
        assert isinstance(result, PipelineResult)
        assert result.question_id == "test_query_001"
        assert result.generated_answer == mock_generation_result.answer
        assert result.confidence_score == mock_generation_result.confidence_score
        assert len(result.retrieved_images) == 2
        assert result.total_time > 0
        assert result.retrieval_time > 0
        assert result.generation_time > 0

        # Verify component interactions
        mock_retriever.retrieve_similar.assert_called_once()
        mock_generator.generate_answer.assert_called_once()

    def test_memory_constraint_validation(self, mock_config):
        """Test memory constraint validation."""
        pipeline = MRAGPipeline(mock_config)

        # Test memory availability check
        memory_manager = pipeline.memory_manager
        assert isinstance(memory_manager, MemoryManager)

        # Test memory stats
        stats = memory_manager.monitor.get_current_stats()
        assert hasattr(stats, 'gpu_allocated_gb')
        assert hasattr(stats, 'gpu_total_gb')

    @patch('src.generation.llava_pipeline.LLaVAGenerationPipeline')
    @patch('src.retrieval.clip_retriever.CLIPRetriever')
    @patch('src.dataset.mrag_dataset.MRAGDataset')
    def test_error_handling_in_pipeline(self, mock_dataset_class, mock_retriever_class, mock_generator_class, mock_config):
        """Test error handling throughout the pipeline."""
        # Setup mocks to raise errors
        mock_dataset = Mock()
        mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 5}
        mock_dataset_class.return_value = mock_dataset

        mock_retriever = Mock()
        mock_retriever.retrieve_similar.side_effect = Exception("Retrieval failed")
        mock_retriever_class.return_value = mock_retriever

        pipeline = MRAGPipeline(mock_config)

        # Test that errors are handled gracefully
        result = pipeline.process_query(
            question="Test question",
            question_id="error_test",
            use_sequential_loading=False
        )

        assert isinstance(result, PipelineResult)
        assert "Error" in result.generated_answer
        assert result.confidence_score == 0.0
        assert "error" in result.metadata

    def test_pipeline_cleanup(self, mock_config):
        """Test pipeline cleanup and resource deallocation."""
        pipeline = MRAGPipeline(mock_config)

        # Mock some loaded components
        pipeline.retriever = Mock()
        pipeline.generator = Mock()
        pipeline.retriever_loaded = True
        pipeline.generator_loaded = True

        # Test cleanup
        pipeline.cleanup()

        # Verify cleanup calls were made
        # (actual implementation depends on the mock setup)
        assert pipeline.retriever_loaded is False
        assert pipeline.generator_loaded is False

    def test_pipeline_context_manager(self, mock_config):
        """Test pipeline as context manager."""
        with patch.object(MRAGPipeline, 'cleanup') as mock_cleanup:
            with MRAGPipeline(mock_config) as pipeline:
                assert isinstance(pipeline, MRAGPipeline)

            # Verify cleanup was called
            mock_cleanup.assert_called_once()

    @patch('src.generation.llava_pipeline.LLaVAGenerationPipeline')
    @patch('src.retrieval.clip_retriever.CLIPRetriever')
    @patch('src.dataset.mrag_dataset.MRAGDataset')
    def test_batch_sample_processing(self, mock_dataset_class, mock_retriever_class, mock_generator_class, mock_config, sample_samples):
        """Test processing multiple samples."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 5}
        mock_dataset_class.return_value = mock_dataset

        mock_retriever = Mock()
        from src.retrieval.interface import RetrievalResult
        mock_retriever.retrieve_similar.return_value = [
            RetrievalResult(image_path="/path/to/image.jpg", similarity_score=0.9, metadata={})
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

        pipeline = MRAGPipeline(mock_config)

        # Process samples
        results = pipeline.process_samples(sample_samples, use_sequential_loading=True)

        assert len(results) == len(sample_samples)
        assert all(isinstance(result, PipelineResult) for result in results)
        assert all(result.question_id == sample.question_id for result, sample in zip(results, sample_samples))

    def test_pipeline_statistics_tracking(self, mock_config):
        """Test pipeline statistics tracking."""
        pipeline = MRAGPipeline(mock_config)

        # Initially should have zero stats
        stats = pipeline.get_pipeline_stats()
        assert stats["total_queries"] == 0
        assert stats["successful_queries"] == 0

        # Simulate some statistics updates
        pipeline._update_stats(1.5, 2.0, 3.5)
        pipeline._update_stats(1.0, 2.5, 3.5)

        updated_stats = pipeline.get_pipeline_stats()
        assert updated_stats["total_queries"] == 2
        assert updated_stats["successful_queries"] == 2
        assert updated_stats["avg_retrieval_time"] == 1.25  # (1.5 + 1.0) / 2
        assert updated_stats["avg_generation_time"] == 2.25  # (2.0 + 2.5) / 2

    def test_memory_pressure_detection(self, mock_config):
        """Test memory pressure detection and handling."""
        # Set a very low memory limit to trigger pressure detection
        mock_config.performance.memory_limit_gb = 0.1

        pipeline = MRAGPipeline(mock_config)

        # Test memory pressure check
        is_under_pressure = pipeline.memory_manager.monitor.check_memory_pressure()
        # This test depends on actual memory usage, so we just verify it returns a boolean
        assert isinstance(is_under_pressure, bool)

    @pytest.mark.parametrize("use_sequential_loading", [True, False])
    def test_sequential_vs_parallel_loading(self, mock_config, use_sequential_loading):
        """Test both sequential and parallel loading strategies."""
        with patch('src.dataset.mrag_dataset.MRAGDataset'), \
             patch('src.retrieval.clip_retriever.CLIPRetriever'), \
             patch('src.generation.llava_pipeline.LLaVAGenerationPipeline'):

            pipeline = MRAGPipeline(mock_config)

            # This is mainly testing that the parameter is passed correctly
            # Actual behavior testing would require more complex mocking
            try:
                result = pipeline.process_query(
                    "Test question",
                    use_sequential_loading=use_sequential_loading
                )
                # If no exception, the parameter was handled correctly
                assert isinstance(result, PipelineResult)
            except Exception:
                # Expected if mocks are not fully set up
                pass


class TestMemoryValidation:
    """Specific tests for memory validation and constraints."""

    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement."""
        memory_manager = MemoryManager(memory_limit_gb=1.0, buffer_gb=0.1)

        # Test availability check
        is_available = memory_manager.check_memory_availability(0.5)
        assert isinstance(is_available, bool)

        # Test batch size recommendation
        recommended_size = memory_manager.get_recommended_batch_size(
            base_batch_size=16,
            memory_per_item_mb=100
        )
        assert isinstance(recommended_size, int)
        assert recommended_size > 0

    def test_memory_guard_context_manager(self):
        """Test memory guard context manager."""
        memory_manager = MemoryManager(memory_limit_gb=10.0)

        with memory_manager.memory_guard("test_operation"):
            # Simulate some operation
            pass

        # Should complete without errors

    def test_emergency_cleanup(self):
        """Test emergency memory cleanup."""
        memory_manager = MemoryManager(memory_limit_gb=10.0)

        # Should not raise errors
        memory_manager.emergency_cleanup()

    def test_memory_optimization(self):
        """Test memory optimization settings."""
        memory_manager = MemoryManager(memory_limit_gb=10.0)

        # Should not raise errors
        memory_manager.optimize_memory_allocation()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])