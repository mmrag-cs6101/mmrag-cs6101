"""
Integration Tests for MRAG-Bench Evaluation Pipeline

End-to-end tests for the complete evaluation system including orchestration,
optimization, and full pipeline integration.
"""

import pytest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.evaluation.orchestrator import (
    EvaluationOrchestrator,
    OptimizationTarget,
    OrchestrationResult
)
from src.evaluation.evaluator import MRAGBenchEvaluator, EvaluationSession
from src.evaluation.optimizer import PerformanceOptimizer, OptimizationConfig
from src.config import MRAGConfig, DatasetConfig, ModelConfig, RetrievalConfig, GenerationConfig, PerformanceConfig
from src.dataset import Sample
from src.pipeline import PipelineResult


class TestEndToEndEvaluation:
    """Integration tests for complete evaluation pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_config(self):
        """Create mock MRAG configuration for testing."""
        return MRAGConfig(
            dataset=DatasetConfig(
                data_path="/mock/data",
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
    def optimization_target(self):
        """Create optimization target for testing."""
        return OptimizationTarget(
            min_accuracy=0.53,
            max_accuracy=0.59,
            max_processing_time=30.0,
            memory_limit_gb=15.0,
            min_samples_per_scenario=5
        )

    @pytest.fixture
    def mock_dataset_with_scenarios(self):
        """Create comprehensive mock dataset with all scenarios."""
        dataset = Mock()

        # Create samples for all perspective change scenarios
        scenarios = ["angle", "partial", "scope", "occlusion"]
        all_samples = {}

        for scenario in scenarios:
            samples = []
            for i in range(10):  # 10 samples per scenario
                sample = Sample(
                    question_id=f"{scenario}_{i:03d}",
                    question=f"What medical condition is shown in this {scenario} view?",
                    image_path=f"/mock/images/{scenario}_{i:03d}.jpg",
                    image=None,
                    ground_truth=f"diagnosis_{i % 3}",  # Cycle through 3 different diagnoses
                    perspective_type=scenario,
                    metadata={"category": f"{scenario}_change", "difficulty": "medium"}
                )
                samples.append(sample)
            all_samples[scenario] = samples

        def get_samples_by_scenario(scenario_type):
            return all_samples.get(scenario_type, [])

        dataset.get_samples_by_scenario = get_samples_by_scenario
        dataset.validate_dataset.return_value = {
            "status": "success",
            "total_samples": 40,  # 10 per scenario * 4 scenarios
            "errors": []
        }

        return dataset

    @pytest.fixture
    def mock_pipeline_with_variable_accuracy(self):
        """Create mock pipeline with variable accuracy for testing optimization."""
        pipeline = Mock()

        def process_query(question, question_id, ground_truth, **kwargs):
            # Simulate different accuracy based on configuration and question
            base_accuracy = 0.5

            # Extract scenario type from question_id
            scenario = question_id.split('_')[0] if '_' in question_id else 'unknown'

            # Simulate scenario-specific performance
            scenario_modifiers = {
                'angle': 0.1,    # Better on angle changes
                'partial': -0.05, # Worse on partial views
                'scope': 0.05,   # Slightly better on scope
                'occlusion': -0.1 # Worse on occlusion
            }

            accuracy_modifier = scenario_modifiers.get(scenario, 0.0)

            # Add some randomness based on question ID
            question_num = int(question_id.split('_')[-1]) if '_' in question_id else 0
            random_modifier = (question_num % 10) * 0.01  # 0-9% variation

            final_accuracy = base_accuracy + accuracy_modifier + random_modifier

            # Determine if answer is correct based on accuracy
            is_correct = np.random.random() < final_accuracy

            if is_correct:
                generated_answer = ground_truth
                confidence = min(0.9, final_accuracy + 0.2)
            else:
                generated_answer = "incorrect_diagnosis"
                confidence = max(0.1, final_accuracy - 0.2)

            processing_time = np.random.uniform(20.0, 35.0)  # Variable processing time

            return PipelineResult(
                question_id=question_id,
                question=question,
                retrieved_images=[f"/mock/retrieved_{question_id}.jpg"],
                retrieval_scores=[np.random.uniform(0.6, 0.9)],
                generated_answer=generated_answer,
                confidence_score=confidence,
                total_time=processing_time,
                retrieval_time=processing_time * 0.3,
                generation_time=processing_time * 0.7,
                memory_usage={"gpu_allocated_gb": np.random.uniform(4.0, 6.0)},
                metadata={"ground_truth": ground_truth, "accuracy_modifier": accuracy_modifier}
            )

        pipeline.process_query.side_effect = process_query
        pipeline.initialize_dataset.return_value = None
        pipeline.cleanup.return_value = None

        return pipeline

    @patch('src.evaluation.orchestrator.MRAGDataset')
    @patch('src.evaluation.orchestrator.MRAGPipeline')
    def test_basic_orchestration_workflow(
        self,
        mock_pipeline_class,
        mock_dataset_class,
        mock_config,
        optimization_target,
        temp_dir,
        mock_dataset_with_scenarios,
        mock_pipeline_with_variable_accuracy
    ):
        """Test basic orchestration workflow."""
        # Setup mocks
        mock_dataset_class.return_value = mock_dataset_with_scenarios
        mock_pipeline_class.return_value = mock_pipeline_with_variable_accuracy

        # Create orchestrator
        orchestrator = EvaluationOrchestrator(
            base_config=mock_config,
            target=optimization_target,
            output_dir=temp_dir
        )

        # Run comprehensive evaluation with minimal optimization
        result = orchestrator.run_comprehensive_evaluation(
            max_optimization_rounds=2,
            early_stopping_patience=1,
            parallel_configs=1
        )

        # Validate result
        assert isinstance(result, OrchestrationResult)
        assert result.session_id is not None
        assert 0.0 <= result.final_accuracy <= 1.0
        assert result.total_optimization_time > 0
        assert isinstance(result.final_evaluation_session, EvaluationSession)
        assert len(result.recommendations) > 0

        # Check that results were saved
        output_path = Path(temp_dir)
        json_files = list(output_path.glob("*_orchestration.json"))
        assert len(json_files) > 0

        summary_files = list(output_path.glob("*_summary.txt"))
        assert len(summary_files) > 0

    @patch('src.evaluation.evaluator.MRAGDataset')
    @patch('src.evaluation.evaluator.MRAGPipeline')
    def test_evaluator_with_all_scenarios(
        self,
        mock_pipeline_class,
        mock_dataset_class,
        mock_config,
        temp_dir,
        mock_dataset_with_scenarios,
        mock_pipeline_with_variable_accuracy
    ):
        """Test evaluator with all perspective change scenarios."""
        # Setup mocks
        mock_dataset_class.return_value = mock_dataset_with_scenarios
        mock_pipeline_class.return_value = mock_pipeline_with_variable_accuracy

        # Create evaluator
        evaluator = MRAGBenchEvaluator(mock_config, temp_dir)

        # Run evaluation on all scenarios
        session = evaluator.evaluate_all_scenarios(max_samples_per_scenario=5)

        # Validate session results
        assert isinstance(session, EvaluationSession)
        assert session.overall_accuracy >= 0.0
        assert session.total_questions > 0

        # Check scenario results
        expected_scenarios = ["angle", "partial", "scope", "occlusion"]
        for scenario in expected_scenarios:
            if scenario in session.scenario_results:
                metrics = session.scenario_results[scenario]
                assert metrics.total_questions > 0
                assert 0.0 <= metrics.accuracy <= 1.0
                assert metrics.avg_processing_time > 0

        # Validate error analysis
        assert "total_errors" in session.error_analysis
        assert session.error_analysis["total_errors"] >= 0

    def test_optimization_target_achievement_detection(
        self,
        mock_config,
        optimization_target,
        temp_dir
    ):
        """Test detection of optimization target achievement."""
        with patch('src.evaluation.orchestrator.MRAGDataset') as mock_dataset_class, \
             patch('src.evaluation.orchestrator.MRAGPipeline') as mock_pipeline_class:

            # Create mock that achieves target immediately
            mock_dataset = Mock()
            mock_dataset.get_samples_by_scenario.return_value = [
                Sample("test_001", "Test question", "/mock/img.jpg", None, "test_answer", "angle", {})
            ]
            mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 1}

            mock_pipeline = Mock()
            mock_pipeline.process_query.return_value = PipelineResult(
                question_id="test_001",
                question="Test question",
                retrieved_images=[],
                retrieval_scores=[],
                generated_answer="test_answer",  # Correct answer
                confidence_score=0.9,
                total_time=20.0,
                retrieval_time=5.0,
                generation_time=15.0,
                memory_usage={"gpu_allocated_gb": 5.0},
                metadata={}
            )

            mock_dataset_class.return_value = mock_dataset
            mock_pipeline_class.return_value = mock_pipeline

            orchestrator = EvaluationOrchestrator(
                base_config=mock_config,
                target=optimization_target,
                output_dir=temp_dir
            )

            # Should detect target achievement quickly
            result = orchestrator.run_comprehensive_evaluation(
                max_optimization_rounds=1,
                early_stopping_patience=1,
                parallel_configs=1
            )

            # Should indicate target was achieved
            assert result.target_achieved == True
            assert optimization_target.min_accuracy <= result.final_accuracy <= optimization_target.max_accuracy

    def test_optimization_convergence(self, mock_config, temp_dir):
        """Test optimization convergence behavior."""
        target = OptimizationTarget(
            min_accuracy=0.8,  # High target that requires optimization
            max_accuracy=0.9,
            min_samples_per_scenario=2
        )

        with patch('src.evaluation.orchestrator.MRAGDataset') as mock_dataset_class, \
             patch('src.evaluation.orchestrator.MRAGPipeline') as mock_pipeline_class:

            # Create mocks that improve with optimization
            mock_dataset = Mock()
            mock_dataset.get_samples_by_scenario.return_value = [
                Sample("test_001", "Test", "/mock/img1.jpg", None, "answer1", "angle", {}),
                Sample("test_002", "Test", "/mock/img2.jpg", None, "answer2", "angle", {})
            ]
            mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 2}

            # Mock pipeline that improves accuracy with certain configurations
            def improving_process_query(question, question_id, ground_truth, **kwargs):
                # Simulate improvement based on configuration
                # (This would normally be handled by the orchestrator's config updates)
                base_accuracy = 0.6
                if question_id == "test_001":
                    generated_answer = ground_truth if np.random.random() < base_accuracy else "wrong"
                else:
                    generated_answer = ground_truth if np.random.random() < base_accuracy else "wrong"

                return PipelineResult(
                    question_id=question_id,
                    question=question,
                    retrieved_images=[],
                    retrieval_scores=[],
                    generated_answer=generated_answer,
                    confidence_score=0.8,
                    total_time=25.0,
                    retrieval_time=5.0,
                    generation_time=20.0,
                    memory_usage={"gpu_allocated_gb": 5.0},
                    metadata={}
                )

            mock_pipeline = Mock()
            mock_pipeline.process_query.side_effect = improving_process_query

            mock_dataset_class.return_value = mock_dataset
            mock_pipeline_class.return_value = mock_pipeline

            orchestrator = EvaluationOrchestrator(
                base_config=mock_config,
                target=target,
                output_dir=temp_dir
            )

            result = orchestrator.run_comprehensive_evaluation(
                max_optimization_rounds=3,
                early_stopping_patience=2,
                parallel_configs=2
            )

            # Should have attempted optimization
            assert len(result.optimization_history) > 0
            assert result.total_optimization_time > 0

    def test_memory_constraint_handling(self, mock_config, optimization_target, temp_dir):
        """Test handling of memory constraints during evaluation."""
        # Set very low memory limit
        optimization_target.memory_limit_gb = 8.0

        with patch('src.evaluation.orchestrator.MRAGDataset') as mock_dataset_class, \
             patch('src.evaluation.orchestrator.MRAGPipeline') as mock_pipeline_class:

            mock_dataset = Mock()
            mock_dataset.get_samples_by_scenario.return_value = [
                Sample("mem_001", "Test", "/mock/img.jpg", None, "answer", "angle", {})
            ]
            mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 1}

            # Mock pipeline that reports high memory usage
            mock_pipeline = Mock()
            mock_pipeline.process_query.return_value = PipelineResult(
                question_id="mem_001",
                question="Test",
                retrieved_images=[],
                retrieval_scores=[],
                generated_answer="answer",
                confidence_score=0.8,
                total_time=25.0,
                retrieval_time=5.0,
                generation_time=20.0,
                memory_usage={"gpu_allocated_gb": 9.0},  # Above limit
                metadata={}
            )

            mock_dataset_class.return_value = mock_dataset
            mock_pipeline_class.return_value = mock_pipeline

            orchestrator = EvaluationOrchestrator(
                base_config=mock_config,
                target=optimization_target,
                output_dir=temp_dir
            )

            result = orchestrator.run_comprehensive_evaluation(
                max_optimization_rounds=1,
                early_stopping_patience=1,
                parallel_configs=1
            )

            # Should include memory-related recommendations
            memory_recommendations = [
                rec for rec in result.recommendations
                if "memory" in rec.lower() or "Memory" in rec
            ]
            assert len(memory_recommendations) > 0

    def test_error_handling_and_recovery(self, mock_config, optimization_target, temp_dir):
        """Test error handling and recovery during evaluation."""
        with patch('src.evaluation.orchestrator.MRAGDataset') as mock_dataset_class, \
             patch('src.evaluation.orchestrator.MRAGPipeline') as mock_pipeline_class:

            mock_dataset = Mock()
            mock_dataset.get_samples_by_scenario.return_value = [
                Sample("error_001", "Test", "/mock/img.jpg", None, "answer", "angle", {})
            ]
            mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 1}

            # Mock pipeline that sometimes fails
            call_count = 0
            def failing_process_query(question, question_id, ground_truth, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Simulated pipeline failure")
                else:
                    return PipelineResult(
                        question_id=question_id,
                        question=question,
                        retrieved_images=[],
                        retrieval_scores=[],
                        generated_answer="answer",
                        confidence_score=0.8,
                        total_time=25.0,
                        retrieval_time=5.0,
                        generation_time=20.0,
                        memory_usage={"gpu_allocated_gb": 5.0},
                        metadata={}
                    )

            mock_pipeline = Mock()
            mock_pipeline.process_query.side_effect = failing_process_query

            mock_dataset_class.return_value = mock_dataset
            mock_pipeline_class.return_value = mock_pipeline

            orchestrator = EvaluationOrchestrator(
                base_config=mock_config,
                target=optimization_target,
                output_dir=temp_dir
            )

            # Should handle errors gracefully and continue
            result = orchestrator.run_comprehensive_evaluation(
                max_optimization_rounds=2,
                early_stopping_patience=1,
                parallel_configs=1
            )

            # Should complete despite initial failure
            assert isinstance(result, OrchestrationResult)

    def test_results_persistence_and_loading(self, mock_config, optimization_target, temp_dir):
        """Test that evaluation results are properly saved and can be loaded."""
        with patch('src.evaluation.orchestrator.MRAGDataset') as mock_dataset_class, \
             patch('src.evaluation.orchestrator.MRAGPipeline') as mock_pipeline_class:

            # Setup minimal mocks
            mock_dataset = Mock()
            mock_dataset.get_samples_by_scenario.return_value = [
                Sample("persist_001", "Test", "/mock/img.jpg", None, "answer", "angle", {})
            ]
            mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 1}

            mock_pipeline = Mock()
            mock_pipeline.process_query.return_value = PipelineResult(
                question_id="persist_001",
                question="Test",
                retrieved_images=[],
                retrieval_scores=[],
                generated_answer="answer",
                confidence_score=0.8,
                total_time=25.0,
                retrieval_time=5.0,
                generation_time=20.0,
                memory_usage={"gpu_allocated_gb": 5.0},
                metadata={}
            )

            mock_dataset_class.return_value = mock_dataset
            mock_pipeline_class.return_value = mock_pipeline

            orchestrator = EvaluationOrchestrator(
                base_config=mock_config,
                target=optimization_target,
                output_dir=temp_dir
            )

            result = orchestrator.run_comprehensive_evaluation(
                max_optimization_rounds=1,
                early_stopping_patience=1,
                parallel_configs=1
            )

            # Verify files were created
            output_path = Path(temp_dir)
            json_files = list(output_path.glob("*_orchestration.json"))
            assert len(json_files) >= 1

            # Verify JSON file can be loaded
            with open(json_files[0], 'r') as f:
                loaded_data = json.load(f)
                assert "session_id" in loaded_data
                assert "final_accuracy" in loaded_data
                assert "target_achieved" in loaded_data

            # Verify summary file exists and has content
            summary_files = list(output_path.glob("*_summary.txt"))
            assert len(summary_files) >= 1

            with open(summary_files[0], 'r') as f:
                summary_content = f.read()
                assert "MRAG-Bench Evaluation Orchestration Summary" in summary_content
                assert result.session_id in summary_content

    def test_cleanup_and_resource_management(self, mock_config, optimization_target, temp_dir):
        """Test proper cleanup and resource management."""
        with patch('src.evaluation.orchestrator.MRAGDataset') as mock_dataset_class, \
             patch('src.evaluation.orchestrator.MRAGPipeline') as mock_pipeline_class:

            mock_dataset = Mock()
            mock_pipeline = Mock()
            mock_dataset_class.return_value = mock_dataset
            mock_pipeline_class.return_value = mock_pipeline

            # Test context manager usage
            with EvaluationOrchestrator(mock_config, optimization_target, temp_dir) as orchestrator:
                assert orchestrator is not None

            # Test manual cleanup
            orchestrator = EvaluationOrchestrator(mock_config, optimization_target, temp_dir)
            orchestrator.evaluator = Mock()
            orchestrator.cleanup()

            # Verify cleanup was called
            orchestrator.evaluator.cleanup.assert_called_once()

    def test_recommendation_generation(self, mock_config, optimization_target, temp_dir):
        """Test recommendation generation based on evaluation results."""
        with patch('src.evaluation.orchestrator.MRAGDataset') as mock_dataset_class, \
             patch('src.evaluation.orchestrator.MRAGPipeline') as mock_pipeline_class:

            mock_dataset = Mock()
            mock_dataset.get_samples_by_scenario.return_value = [
                Sample("rec_001", "Test", "/mock/img.jpg", None, "answer", "angle", {})
            ]
            mock_dataset.validate_dataset.return_value = {"status": "success", "total_samples": 1}

            # Mock pipeline with specific characteristics for recommendation testing
            mock_pipeline = Mock()
            mock_pipeline.process_query.return_value = PipelineResult(
                question_id="rec_001",
                question="Test",
                retrieved_images=[],
                retrieval_scores=[],
                generated_answer="wrong_answer",  # Incorrect for low accuracy
                confidence_score=0.3,
                total_time=40.0,  # Slow processing
                retrieval_time=10.0,
                generation_time=30.0,
                memory_usage={"gpu_allocated_gb": 14.5},  # Near memory limit
                metadata={}
            )

            mock_dataset_class.return_value = mock_dataset
            mock_pipeline_class.return_value = mock_pipeline

            # Set target for triggering recommendations
            optimization_target.min_accuracy = 0.8  # High target to trigger accuracy recommendations
            optimization_target.max_processing_time = 25.0  # Low limit to trigger speed recommendations

            orchestrator = EvaluationOrchestrator(
                base_config=mock_config,
                target=optimization_target,
                output_dir=temp_dir
            )

            result = orchestrator.run_comprehensive_evaluation(
                max_optimization_rounds=1,
                early_stopping_patience=1,
                parallel_configs=1
            )

            # Should have generated relevant recommendations
            assert len(result.recommendations) > 0

            # Check for specific types of recommendations
            accuracy_recs = [rec for rec in result.recommendations if "accuracy" in rec.lower()]
            speed_recs = [rec for rec in result.recommendations if "processing time" in rec.lower() or "time" in rec.lower()]
            memory_recs = [rec for rec in result.recommendations if "memory" in rec.lower()]

            # Should have at least one type of recommendation
            assert len(accuracy_recs) > 0 or len(speed_recs) > 0 or len(memory_recs) > 0