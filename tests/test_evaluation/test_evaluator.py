"""
Unit Tests for MRAG-Bench Evaluator

Comprehensive tests for the evaluation framework including accuracy calculation,
scenario filtering, and metrics computation.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.evaluation.evaluator import (
    MRAGBenchEvaluator,
    PerspectiveChangeType,
    ScenarioMetrics,
    EvaluationSession
)
from src.config import MRAGConfig, DatasetConfig, ModelConfig, RetrievalConfig, GenerationConfig, PerformanceConfig
from src.dataset import Sample
from src.pipeline import PipelineResult


class TestMRAGBenchEvaluator:
    """Test cases for MRAGBenchEvaluator."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_config(self):
        """Create mock MRAG configuration."""
        return MRAGConfig(
            dataset=DatasetConfig(
                data_path="/mock/path",
                batch_size=4,
                image_size=(224, 224),
                cache_embeddings=False,
                embedding_cache_path="/mock/cache"
            ),
            model=ModelConfig(
                vlm_name="mock-vlm",
                retriever_name="mock-retriever",
                quantization="4bit",
                max_memory_gb=10.0,
                device="cuda"
            ),
            retrieval=RetrievalConfig(
                embedding_dim=512,
                top_k=5,
                similarity_threshold=0.7,
                batch_size=8
            ),
            generation=GenerationConfig(
                max_length=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                top_k=50
            ),
            performance=PerformanceConfig(
                memory_limit_gb=15.0,
                memory_buffer_gb=1.0,
                max_batch_size=8
            )
        )

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset with test samples."""
        dataset = Mock()

        # Mock samples for different scenarios
        samples = {
            "angle": [
                Sample(
                    question_id="angle_001",
                    question="What is the medical condition shown?",
                    image_path="/mock/image1.jpg",
                    image=None,
                    ground_truth="pneumonia",
                    perspective_type="angle",
                    metadata={"category": "angle_change"}
                ),
                Sample(
                    question_id="angle_002",
                    question="Describe the anatomical structure.",
                    image_path="/mock/image2.jpg",
                    image=None,
                    ground_truth="heart ventricle",
                    perspective_type="angle",
                    metadata={"category": "angle_change"}
                )
            ],
            "partial": [
                Sample(
                    question_id="partial_001",
                    question="What organ is partially visible?",
                    image_path="/mock/image3.jpg",
                    image=None,
                    ground_truth="liver",
                    perspective_type="partial",
                    metadata={"category": "partial_view"}
                )
            ]
        }

        def get_samples_by_scenario(scenario_type):
            return samples.get(scenario_type, [])

        dataset.get_samples_by_scenario = get_samples_by_scenario
        dataset.validate_dataset.return_value = {
            "status": "success",
            "total_samples": 3,
            "errors": []
        }

        return dataset

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline with predictable results."""
        pipeline = Mock()

        def process_query(question, question_id, ground_truth, **kwargs):
            # Simulate different accuracy levels based on question_id
            if "angle_001" in question_id:
                generated_answer = "pneumonia"  # Correct
                confidence = 0.9
            elif "angle_002" in question_id:
                generated_answer = "cardiac chamber"  # Partially correct
                confidence = 0.7
            elif "partial_001" in question_id:
                generated_answer = "kidney"  # Incorrect
                confidence = 0.5
            else:
                generated_answer = "unknown"
                confidence = 0.1

            return PipelineResult(
                question_id=question_id,
                question=question,
                retrieved_images=["/mock/retrieved1.jpg"],
                retrieval_scores=[0.8],
                generated_answer=generated_answer,
                confidence_score=confidence,
                total_time=2.5,
                retrieval_time=1.0,
                generation_time=1.5,
                memory_usage={"gpu_allocated_gb": 5.0},
                metadata={"ground_truth": ground_truth}
            )

        pipeline.process_query.side_effect = process_query
        pipeline.initialize_dataset.return_value = None
        pipeline.cleanup.return_value = None

        return pipeline

    def test_evaluator_initialization(self, mock_config, temp_dir):
        """Test evaluator initialization."""
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        assert evaluator.config == mock_config
        assert evaluator.output_dir == Path(temp_dir)
        assert evaluator.dataset is None
        assert evaluator.pipeline is None
        assert evaluator.current_session is None

    @patch('src.evaluation.evaluator.MRAGDataset')
    @patch('src.evaluation.evaluator.MRAGPipeline')
    def test_initialize_components(self, mock_pipeline_class, mock_dataset_class, mock_config, temp_dir):
        """Test component initialization."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 100}
        mock_dataset_class.return_value = mock_dataset

        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)
        evaluator.initialize_components()

        assert evaluator.dataset is not None
        assert evaluator.pipeline is not None
        mock_dataset.validate_dataset.assert_called_once()
        mock_pipeline.initialize_dataset.assert_called_once()

    def test_is_answer_correct(self, mock_config, temp_dir):
        """Test answer correctness evaluation."""
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        # Test exact match
        assert evaluator._is_answer_correct("pneumonia", "pneumonia") == True

        # Test case insensitive
        assert evaluator._is_answer_correct("Pneumonia", "pneumonia") == True

        # Test partial match with medical keywords
        assert evaluator._is_answer_correct("acute pneumonia infection", "pneumonia") == True

        # Test incorrect answer
        assert evaluator._is_answer_correct("kidney disease", "pneumonia") == False

        # Test empty answers
        assert evaluator._is_answer_correct("", "pneumonia") == False
        assert evaluator._is_answer_correct("pneumonia", "") == False

    def test_normalize_answer(self, mock_config, temp_dir):
        """Test answer normalization."""
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        # Test basic normalization
        result = evaluator._normalize_answer("Acute Pneumonia!")
        assert result == "acute pneumonia"

        # Test stop word removal
        result = evaluator._normalize_answer("The patient has a severe pneumonia.")
        assert "patient" in result and "severe" in result and "pneumonia" in result
        assert "the" not in result and "has" not in result

    def test_extract_medical_keywords(self, mock_config, temp_dir):
        """Test medical keyword extraction."""
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        text = "The patient shows signs of acute pneumonia with lung inflammation"
        keywords = evaluator._extract_medical_keywords(text)

        assert "lung" in keywords
        assert "pneumonia" in keywords
        assert "acute" in keywords
        assert "inflammation" in keywords
        assert "patient" not in keywords  # Not in medical terms list

    @patch('src.evaluation.evaluator.MRAGDataset')
    @patch('src.evaluation.evaluator.MRAGPipeline')
    def test_evaluate_scenario(self, mock_pipeline_class, mock_dataset_class, mock_config, temp_dir, mock_dataset, mock_pipeline):
        """Test single scenario evaluation."""
        # Setup mocks
        mock_dataset_class.return_value = mock_dataset
        mock_pipeline_class.return_value = mock_pipeline

        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)
        evaluator.dataset = mock_dataset
        evaluator.pipeline = mock_pipeline

        # Test angle scenario evaluation
        metrics = evaluator.evaluate_scenario(PerspectiveChangeType.ANGLE, max_samples=2)

        assert isinstance(metrics, ScenarioMetrics)
        assert metrics.scenario_type == "angle"
        assert metrics.total_questions == 2
        assert metrics.correct_answers >= 0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.avg_processing_time > 0
        assert len(metrics.confidence_scores) == 2

    @patch('src.evaluation.evaluator.MRAGDataset')
    @patch('src.evaluation.evaluator.MRAGPipeline')
    def test_evaluate_all_scenarios(self, mock_pipeline_class, mock_dataset_class, mock_config, temp_dir, mock_dataset, mock_pipeline):
        """Test comprehensive evaluation of all scenarios."""
        # Setup mocks
        mock_dataset_class.return_value = mock_dataset
        mock_pipeline_class.return_value = mock_pipeline

        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)
        evaluator.dataset = mock_dataset
        evaluator.pipeline = mock_pipeline

        # Test comprehensive evaluation
        session = evaluator.evaluate_all_scenarios(max_samples_per_scenario=2)

        assert isinstance(session, EvaluationSession)
        assert session.session_id is not None
        assert session.timestamp is not None
        assert len(session.scenario_results) <= 4  # Up to 4 scenarios
        assert 0.0 <= session.overall_accuracy <= 1.0
        assert session.total_questions >= 0
        assert session.avg_processing_time > 0

    def test_analyze_errors(self, mock_config, temp_dir):
        """Test error analysis functionality."""
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        # Create mock scenario results with errors
        scenario_results = {
            "angle": ScenarioMetrics(
                scenario_type="angle",
                total_questions=10,
                correct_answers=6,
                accuracy=0.6,
                avg_processing_time=25.0,
                avg_retrieval_time=5.0,
                avg_generation_time=20.0,
                confidence_scores=[0.8, 0.7, 0.9],
                error_count=1,
                error_rate=0.1
            ),
            "partial": ScenarioMetrics(
                scenario_type="partial",
                total_questions=10,
                correct_answers=3,
                accuracy=0.3,
                avg_processing_time=35.0,
                avg_retrieval_time=8.0,
                avg_generation_time=27.0,
                confidence_scores=[0.5, 0.4, 0.6],
                error_count=2,
                error_rate=0.2
            )
        }

        error_analysis = evaluator._analyze_errors(scenario_results)

        assert error_analysis["total_errors"] == 3
        assert "angle" in error_analysis["error_by_scenario"]
        assert "partial" in error_analysis["error_by_scenario"]
        assert len(error_analysis["performance_bottlenecks"]) > 0  # partial scenario is slow
        assert len(error_analysis["common_failure_patterns"]) > 0  # partial scenario has low accuracy

    def test_get_config_summary(self, mock_config, temp_dir):
        """Test configuration summary generation."""
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        config_summary = evaluator._get_config_summary()

        assert "model_names" in config_summary
        assert "vlm" in config_summary["model_names"]
        assert "retriever" in config_summary["model_names"]
        assert "quantization" in config_summary
        assert "memory_limit" in config_summary
        assert "retrieval_config" in config_summary
        assert "generation_config" in config_summary

    def test_evaluation_session_serialization(self, mock_config, temp_dir):
        """Test evaluation session can be serialized to JSON."""
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        # Create a mock session
        scenario_metrics = ScenarioMetrics(
            scenario_type="angle",
            total_questions=10,
            correct_answers=7,
            accuracy=0.7,
            avg_processing_time=20.0,
            avg_retrieval_time=5.0,
            avg_generation_time=15.0,
            confidence_scores=[0.8, 0.9, 0.7],
            error_count=0,
            error_rate=0.0
        )

        session = EvaluationSession(
            session_id="test_session",
            timestamp="2024-01-01 12:00:00",
            config_summary=evaluator._get_config_summary(),
            scenario_results={"angle": scenario_metrics},
            overall_accuracy=0.7,
            total_questions=10,
            total_correct=7,
            avg_processing_time=20.0,
            memory_stats={"gpu_allocated_gb": 5.0},
            error_analysis={"total_errors": 0}
        )

        # Test serialization
        try:
            json_str = json.dumps(session, default=str)
            assert len(json_str) > 0
        except Exception as e:
            pytest.fail(f"Session serialization failed: {e}")

    def test_memory_management_during_evaluation(self, mock_config, temp_dir):
        """Test memory management during evaluation."""
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        # Test memory manager is initialized
        assert evaluator.memory_manager is not None
        assert evaluator.memory_manager.memory_limit_gb == mock_config.performance.memory_limit_gb

    def test_results_caching(self, mock_config, temp_dir, mock_dataset, mock_pipeline):
        """Test evaluation results caching."""
        with patch('src.evaluation.evaluator.MRAGDataset') as mock_dataset_class, \
             patch('src.evaluation.evaluator.MRAGPipeline') as mock_pipeline_class:

            mock_dataset_class.return_value = mock_dataset
            mock_pipeline_class.return_value = mock_pipeline

            evaluator = MRAGBenchEvaluator(mock_config, temp_dir)
            evaluator.dataset = mock_dataset
            evaluator.pipeline = mock_pipeline

            # First evaluation should process samples
            metrics1 = evaluator.evaluate_scenario(PerspectiveChangeType.ANGLE, max_samples=2, use_cache=True)

            # Second evaluation should use cache
            metrics2 = evaluator.evaluate_scenario(PerspectiveChangeType.ANGLE, max_samples=2, use_cache=True)

            # Results should be identical (from cache)
            assert metrics1.accuracy == metrics2.accuracy
            assert metrics1.total_questions == metrics2.total_questions

    def test_cleanup(self, mock_config, temp_dir):
        """Test resource cleanup."""
        with patch('src.evaluation.evaluator.MRAGPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline

            evaluator = MRAGBenchEvaluator(mock_config, temp_dir)
            evaluator.pipeline = mock_pipeline

            # Test cleanup
            evaluator.cleanup()

            # Verify pipeline cleanup was called
            mock_pipeline.cleanup.assert_called_once()

    def test_context_manager(self, mock_config, temp_dir):
        """Test evaluator as context manager."""
        with patch('src.evaluation.evaluator.MRAGPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline

            # Test context manager usage
            with MRAGBenchEvaluator(mock_config, temp_dir) as evaluator:
                evaluator.pipeline = mock_pipeline
                assert evaluator is not None

            # Verify cleanup was called on exit
            mock_pipeline.cleanup.assert_called_once()


class TestPerspectiveChangeType:
    """Test cases for PerspectiveChangeType enum."""

    def test_perspective_change_types(self):
        """Test perspective change type enumeration."""
        assert PerspectiveChangeType.ANGLE.value == "angle"
        assert PerspectiveChangeType.PARTIAL.value == "partial"
        assert PerspectiveChangeType.SCOPE.value == "scope"
        assert PerspectiveChangeType.OCCLUSION.value == "occlusion"

    def test_enum_iteration(self):
        """Test iteration over perspective change types."""
        types = [t.value for t in PerspectiveChangeType]
        expected = ["angle", "partial", "scope", "occlusion"]
        assert types == expected


class TestScenarioMetrics:
    """Test cases for ScenarioMetrics dataclass."""

    def test_scenario_metrics_creation(self):
        """Test scenario metrics creation and access."""
        metrics = ScenarioMetrics(
            scenario_type="angle",
            total_questions=100,
            correct_answers=75,
            accuracy=0.75,
            avg_processing_time=25.5,
            avg_retrieval_time=5.0,
            avg_generation_time=20.5,
            confidence_scores=[0.8, 0.9, 0.7, 0.85],
            error_count=2,
            error_rate=0.02
        )

        assert metrics.scenario_type == "angle"
        assert metrics.total_questions == 100
        assert metrics.correct_answers == 75
        assert metrics.accuracy == 0.75
        assert metrics.avg_processing_time == 25.5
        assert len(metrics.confidence_scores) == 4
        assert metrics.error_count == 2
        assert metrics.error_rate == 0.02

    def test_scenario_metrics_validation(self):
        """Test scenario metrics data validation."""
        metrics = ScenarioMetrics(
            scenario_type="test",
            total_questions=10,
            correct_answers=8,
            accuracy=0.8,
            avg_processing_time=20.0,
            avg_retrieval_time=5.0,
            avg_generation_time=15.0,
            confidence_scores=[0.9, 0.8, 0.7],
            error_count=0,
            error_rate=0.0
        )

        # Validate accuracy calculation
        calculated_accuracy = metrics.correct_answers / metrics.total_questions
        assert abs(calculated_accuracy - metrics.accuracy) < 0.001

        # Validate timing breakdown
        assert metrics.avg_processing_time >= (metrics.avg_retrieval_time + metrics.avg_generation_time) * 0.9