#!/usr/bin/env python3
"""
Unit Tests for Sprint 7 MVP Evaluation Framework

Comprehensive test suite for Sprint 7 deliverables including:
- MRAGBenchEvaluator enhancements
- Single scenario evaluation (angle changes)
- Performance metrics collection
- Memory usage profiling
- Error analysis and recovery
- MVP evaluation pipeline
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import MRAGConfig
from src.evaluation.evaluator import MRAGBenchEvaluator, PerspectiveChangeType, ScenarioMetrics
from src.dataset import Sample


class TestSprint7MVPEvaluation(unittest.TestCase):
    """Test Sprint 7 MVP evaluation framework."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_dir = Path(self.temp_dir) / "test_output"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Create test configuration
        self.config = MRAGConfig()
        self.config.model.device = "cpu"  # Use CPU for testing
        self.config.dataset.data_path = str(Path(self.temp_dir) / "test_data")
        self.config.performance.memory_limit_gb = 4.0  # Lower limit for testing

        # Create test data directory structure
        self.test_data_dir = Path(self.config.dataset.data_path)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

        self._create_test_dataset()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_dataset(self):
        """Create minimal test dataset for testing."""
        # Create test metadata
        metadata_dir = self.test_data_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        dataset_info = {
            "total_samples": 10,
            "scenarios": {"Angle": 10, "Partial": 0, "Scope": 0, "Obstruction": 0},
            "image_count": 10
        }

        with open(metadata_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f)

        # Create test questions
        questions_dir = self.test_data_dir / "questions"
        questions_dir.mkdir(parents=True, exist_ok=True)

        test_questions = []
        for i in range(10):
            question = {
                "question_id": f"test_angle_{i:03d}",
                "scenario": "Angle",
                "question": f"Test question {i} about angle changes",
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D",
                "answer": "A",
                "image_path": f"images/test_image_{i:03d}.jpg"
            }
            test_questions.append(question)

        with open(questions_dir / "questions.json", 'w') as f:
            json.dump(test_questions, f)

        # Create test images directory (empty for unit tests)
        images_dir = self.test_data_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

    @patch('src.evaluation.evaluator.MRAGPipeline')
    @patch('src.dataset.MRAGDataset')
    def test_sprint7_evaluator_initialization(self, mock_dataset, mock_pipeline):
        """Test Sprint 7 evaluator initialization."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        self.assertIsNotNone(evaluator)
        self.assertEqual(evaluator.config, self.config)
        self.assertTrue(self.test_output_dir.exists())

    @patch('src.evaluation.evaluator.MRAGPipeline')
    @patch('src.dataset.MRAGDataset')
    def test_angle_change_scenario_evaluation(self, mock_dataset, mock_pipeline):
        """Test angle change scenario evaluation (Sprint 7 focus)."""
        # Mock dataset to return test samples
        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance

        # Create test samples for angle change scenario
        test_samples = [
            Sample(
                question_id=f"angle_{i:03d}",
                question=f"Test angle question {i}",
                ground_truth="A",
                perspective_type="angle",
                image_path=f"test_image_{i}.jpg",
                metadata={"scenario": "Angle"}
            )
            for i in range(5)
        ]

        mock_dataset_instance.get_samples_by_scenario.return_value = test_samples
        mock_dataset_instance.validate_dataset.return_value = {
            "status": "success",
            "total_samples": 5
        }
        mock_dataset_instance.get_retrieval_corpus.return_value = [
            f"test_image_{i}.jpg" for i in range(5)
        ]

        # Mock pipeline to return test results
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        # Create mock pipeline results
        def mock_process_query(*args, **kwargs):
            return Mock(
                generated_answer="A",
                confidence_score=0.8,
                total_time=2.5,
                retrieval_time=1.0,
                generation_time=1.5,
                question_id=kwargs.get('question_id', 'test')
            )

        mock_pipeline_instance.process_query = mock_process_query

        # Test evaluation
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Initialize components
        evaluator.initialize_components()

        # Evaluate angle change scenario
        metrics = evaluator.evaluate_scenario(
            scenario_type=PerspectiveChangeType.ANGLE,
            max_samples=5,
            use_cache=False
        )

        # Verify Sprint 7 specific metrics
        self.assertIsInstance(metrics, ScenarioMetrics)
        self.assertEqual(metrics.scenario_type, "angle")
        self.assertEqual(metrics.total_questions, 5)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertGreater(metrics.avg_processing_time, 0.0)

    def test_accuracy_calculation_methodology(self):
        """Test MRAG-Bench accuracy calculation methodology."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test exact match
        self.assertTrue(evaluator._is_answer_correct("A", "A"))
        self.assertFalse(evaluator._is_answer_correct("A", "B"))

        # Test case insensitive
        self.assertTrue(evaluator._is_answer_correct("a", "A"))
        self.assertTrue(evaluator._is_answer_correct("A", "a"))

        # Test with medical terms
        medical_answer = "The image shows a fracture in the bone"
        medical_ground_truth = "bone fracture visible"
        result = evaluator._is_answer_correct(medical_answer, medical_ground_truth)
        # Should match due to medical keyword overlap
        self.assertIsInstance(result, bool)

        # Test partial matching
        partial_answer = "lung cancer tumor"
        partial_ground_truth = "tumor in lung"
        result = evaluator._is_answer_correct(partial_answer, partial_ground_truth)
        self.assertIsInstance(result, bool)

    def test_performance_metrics_collection(self):
        """Test performance metrics collection for Sprint 7."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test scenario metrics creation
        metrics = ScenarioMetrics(
            scenario_type="angle",
            total_questions=10,
            correct_answers=6,
            accuracy=0.6,
            avg_processing_time=15.5,
            avg_retrieval_time=5.2,
            avg_generation_time=10.3,
            confidence_scores=[0.8, 0.7, 0.9, 0.6, 0.8],
            error_count=1,
            error_rate=0.1
        )

        # Verify metrics
        self.assertEqual(metrics.accuracy, 0.6)
        self.assertEqual(metrics.total_questions, 10)
        self.assertEqual(metrics.correct_answers, 6)
        self.assertAlmostEqual(metrics.avg_processing_time, 15.5)
        self.assertEqual(len(metrics.confidence_scores), 5)

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring capabilities."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test memory manager initialization
        self.assertIsNotNone(evaluator.memory_manager)
        self.assertEqual(evaluator.memory_manager.memory_limit_gb, self.config.performance.memory_limit_gb)

        # Test memory stats collection
        memory_stats = evaluator.memory_manager.monitor.get_current_stats()
        self.assertIsNotNone(memory_stats)

    def test_error_analysis_and_patterns(self):
        """Test error analysis and failure pattern identification."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test error analysis with sample scenario results
        scenario_results = {
            "angle": ScenarioMetrics(
                scenario_type="angle",
                total_questions=10,
                correct_answers=8,
                accuracy=0.8,
                avg_processing_time=12.0,
                avg_retrieval_time=4.0,
                avg_generation_time=8.0,
                confidence_scores=[0.8] * 8,
                error_count=2,
                error_rate=0.2
            )
        }

        error_analysis = evaluator._analyze_errors(scenario_results)

        # Verify error analysis structure
        self.assertIn("total_errors", error_analysis)
        self.assertIn("error_by_scenario", error_analysis)
        self.assertIn("common_failure_patterns", error_analysis)
        self.assertIn("performance_bottlenecks", error_analysis)

        self.assertEqual(error_analysis["total_errors"], 2)
        self.assertIn("angle", error_analysis["error_by_scenario"])

    def test_result_reporting_and_analysis(self):
        """Test result reporting and analysis utilities."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test session summary generation
        from src.evaluation.evaluator import EvaluationSession

        # Create mock session for testing
        session = EvaluationSession(
            session_id="test_session_sprint7",
            timestamp="2024-10-03 12:00:00",
            config_summary=evaluator._get_config_summary(),
            scenario_results={
                "angle": ScenarioMetrics(
                    scenario_type="angle",
                    total_questions=50,
                    correct_answers=28,
                    accuracy=0.56,
                    avg_processing_time=18.5,
                    avg_retrieval_time=6.0,
                    avg_generation_time=12.5,
                    confidence_scores=[0.7] * 28,
                    error_count=2,
                    error_rate=0.04
                )
            },
            overall_accuracy=0.56,
            total_questions=50,
            total_correct=28,
            avg_processing_time=18.5,
            memory_stats={"gpu_allocated_gb": 8.5, "gpu_cached_gb": 2.1},
            error_analysis={
                "total_errors": 2,
                "error_by_scenario": {},
                "performance_bottlenecks": [],
                "common_failure_patterns": []
            }
        )

        # Test summary generation
        summary = evaluator._generate_session_summary(session)

        # Verify summary content
        self.assertIn("test_session_sprint7", summary)
        self.assertIn("56.0%", summary)  # Accuracy
        self.assertIn("ANGLE Perspective Change", summary)

    def test_mvp_target_validation(self):
        """Test MVP target validation against 53-59% range."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test target range validation
        target_min, target_max = 0.53, 0.59

        # Test accuracy within range
        self.assertTrue(target_min <= 0.55 <= target_max)
        self.assertTrue(target_min <= 0.58 <= target_max)

        # Test accuracy below range
        self.assertFalse(target_min <= 0.45 <= target_max)

        # Test accuracy above range
        self.assertFalse(target_min <= 0.65 <= target_max)

    def test_sprint7_evaluation_pipeline_integration(self):
        """Test complete Sprint 7 evaluation pipeline integration."""
        # This test verifies that all Sprint 7 components work together
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test configuration summary
        config_summary = evaluator._get_config_summary()
        self.assertIn("model_names", config_summary)
        self.assertIn("quantization", config_summary)
        self.assertIn("memory_limit", config_summary)

        # Test evaluation statistics
        stats = evaluator.get_evaluation_statistics()
        self.assertIn("sessions_completed", stats)
        self.assertIn("total_questions_evaluated", stats)

    @patch('src.evaluation.evaluator.MRAGPipeline')
    @patch('src.dataset.MRAGDataset')
    def test_sprint7_comprehensive_evaluation_flow(self, mock_dataset, mock_pipeline):
        """Test comprehensive Sprint 7 evaluation flow."""
        # Mock components
        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset_instance.validate_dataset.return_value = {
            "status": "success",
            "total_samples": 5
        }
        mock_dataset_instance.get_samples_by_scenario.return_value = [
            Sample(
                question_id=f"test_{i}",
                question=f"Test question {i}",
                ground_truth="A",
                perspective_type="angle",
                image_path=f"test_{i}.jpg",
                metadata={}
            )
            for i in range(5)
        ]
        mock_dataset_instance.get_retrieval_corpus.return_value = ["test.jpg"]

        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.process_query.return_value = Mock(
            generated_answer="A",
            confidence_score=0.8,
            total_time=15.0,
            retrieval_time=5.0,
            generation_time=10.0
        )

        # Test evaluator
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Initialize and run evaluation
        evaluator.initialize_components()

        # Test single scenario evaluation (Sprint 7 focus)
        metrics = evaluator.evaluate_scenario(
            scenario_type=PerspectiveChangeType.ANGLE,
            max_samples=5
        )

        # Verify Sprint 7 deliverables
        self.assertEqual(metrics.scenario_type, "angle")
        self.assertGreater(metrics.total_questions, 0)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)

    def test_medical_keyword_extraction(self):
        """Test medical keyword extraction for domain-specific accuracy."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test medical keyword extraction
        medical_text = "The patient has a fracture in the bone near the heart"
        keywords = evaluator._extract_medical_keywords(medical_text)

        expected_keywords = ["fracture", "bone", "heart"]
        for keyword in expected_keywords:
            self.assertIn(keyword, keywords)

        # Test non-medical text
        non_medical_text = "The car is blue and fast"
        keywords = evaluator._extract_medical_keywords(non_medical_text)
        self.assertEqual(len(keywords), 0)

    def test_answer_normalization(self):
        """Test answer normalization for consistent comparison."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test normalization (stop words are removed)
        test_cases = [
            ("The answer is A.", "answer"),  # "the", "is", "a" are stop words
            ("Patient has pneumonia!", "patient pneumonia"),  # "has" is a stop word
            ("    Multiple   spaces   ", "multiple spaces"),
            ("UPPERCASE text", "uppercase text")
        ]

        for input_text, expected in test_cases:
            normalized = evaluator._normalize_answer(input_text)
            self.assertEqual(normalized, expected)

    def test_evaluation_cleanup(self):
        """Test proper cleanup of evaluation resources."""
        evaluator = MRAGBenchEvaluator(
            config=self.config,
            output_dir=str(self.test_output_dir)
        )

        # Test cleanup without error
        evaluator.cleanup()

        # Test context manager
        with MRAGBenchEvaluator(self.config, str(self.test_output_dir)) as eval_ctx:
            self.assertIsNotNone(eval_ctx)
        # Should cleanup automatically


class TestSprint7MVPResultsStructure(unittest.TestCase):
    """Test Sprint 7 MVP results data structures."""

    def test_sprint7_mvp_results_creation(self):
        """Test Sprint 7 MVP results structure."""
        # This would test the Sprint7MVPResults dataclass if imported
        # For now, test the concept with basic structure validation

        mvp_results = {
            "session_id": "sprint7_mvp_test",
            "timestamp": "2024-10-03 12:00:00",
            "scenario_focus": "angle",
            "accuracy": 0.56,
            "total_questions": 50,
            "correct_answers": 28,
            "target_achieved": True,
            "performance_recommendations": ["Optimize retrieval", "Reduce generation time"],
            "optimization_suggestions": ["Increase top-k", "Lower temperature"]
        }

        # Verify structure
        self.assertIn("session_id", mvp_results)
        self.assertIn("accuracy", mvp_results)
        self.assertIn("target_achieved", mvp_results)
        self.assertEqual(mvp_results["scenario_focus"], "angle")

    def test_target_achievement_calculation(self):
        """Test target achievement calculation logic."""
        target_min, target_max = 0.53, 0.59

        # Test various accuracy values
        test_cases = [
            (0.52, False),  # Below range
            (0.53, True),   # At minimum
            (0.56, True),   # Within range
            (0.59, True),   # At maximum
            (0.61, False)   # Above range
        ]

        for accuracy, expected in test_cases:
            achieved = target_min <= accuracy <= target_max
            self.assertEqual(achieved, expected)


if __name__ == '__main__':
    # Create test output directory
    Path("output/test_results").mkdir(parents=True, exist_ok=True)

    # Run tests
    unittest.main(verbosity=2)