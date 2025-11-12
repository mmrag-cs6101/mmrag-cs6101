"""
Unit Tests for MRAG Dataset Implementation

Comprehensive test suite for MRAGDataset class and related functionality.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('/mnt/d/dev/mmrag-cs6101/src')

from dataset.interface import Sample, BatchData
from dataset.mrag_dataset import MRAGDataset
from dataset.preprocessing import ImagePreprocessor, PreprocessingConfig
from dataset.data_loader import MemoryAwareDataLoader, StreamingConfig
from dataset.validation import DatasetValidator, ValidationResult
from utils.memory_manager import MemoryManager


class TestMRAGDataset:
    """Test suite for MRAGDataset class."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "mrag_bench"

            # Create directory structure
            (dataset_path / "images").mkdir(parents=True)
            (dataset_path / "questions").mkdir(parents=True)
            (dataset_path / "metadata").mkdir(parents=True)
            (dataset_path / "annotations").mkdir(parents=True)

            # Create test images
            for i in range(5):
                img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                img.save(dataset_path / "images" / f"image_{i:06d}.jpg")

            # Create test questions data
            questions_data = [
                {
                    "question_id": f"mrag_{i:06d}",
                    "question": f"What is shown in this medical image {i}?",
                    "choices": [f"Option A{i}", f"Option B{i}", f"Option C{i}", f"Option D{i}"],
                    "answer": f"Option A{i}",
                    "image_path": f"images/image_{i:06d}.jpg",
                    "category": ["angle_change", "partial_view", "scope_variation", "occlusion"][i % 4],
                    "scenario": ["angle_change", "partial_view", "scope_variation", "occlusion"][i % 4]
                }
                for i in range(5)
            ]

            with open(dataset_path / "questions" / "questions.json", 'w') as f:
                json.dump(questions_data, f)

            # Create metadata
            metadata = {
                "total_samples": 5,
                "features": ["question", "choices", "answer", "image", "category"],
                "scenarios": {
                    "angle_change": 2,
                    "partial_view": 1,
                    "scope_variation": 1,
                    "occlusion": 1
                },
                "image_count": 5
            }

            with open(dataset_path / "metadata" / "dataset_info.json", 'w') as f:
                json.dump(metadata, f)

            # Create scenario mapping
            scenario_mapping = {
                scenario: {
                    "count": count,
                    "samples": [q for q in questions_data if q['scenario'] == scenario]
                }
                for scenario, count in metadata["scenarios"].items()
            }

            with open(dataset_path / "metadata" / "scenario_mapping.json", 'w') as f:
                json.dump(scenario_mapping, f)

            yield str(dataset_path)

    @pytest.fixture
    def mrag_dataset(self, temp_dataset_dir):
        """Create MRAGDataset instance for testing."""
        return MRAGDataset(temp_dataset_dir, batch_size=2)

    def test_dataset_initialization(self, mrag_dataset):
        """Test dataset initialization."""
        assert mrag_dataset.data_path.exists()
        assert mrag_dataset.batch_size == 2
        assert mrag_dataset.total_samples == 5
        assert len(mrag_dataset.perspective_types) == 4

    def test_load_metadata(self, mrag_dataset):
        """Test metadata loading."""
        assert len(mrag_dataset.questions_data) == 5
        assert mrag_dataset.metadata["total_samples"] == 5
        assert len(mrag_dataset.scenario_mapping) > 0

    def test_perspective_scenario_mapping(self, mrag_dataset):
        """Test perspective change scenario mapping."""
        assert 'angle_change' in mrag_dataset.category_to_perspective
        assert 'partial_view' in mrag_dataset.category_to_perspective
        assert 'scope_variation' in mrag_dataset.category_to_perspective
        assert 'occlusion' in mrag_dataset.category_to_perspective

        # Check that mappings are to valid perspective types
        for category, perspective in mrag_dataset.category_to_perspective.items():
            assert perspective in mrag_dataset.perspective_types

    def test_load_scenario(self, mrag_dataset):
        """Test scenario-specific sample loading."""
        # Test each perspective type
        for perspective_type in mrag_dataset.perspective_types:
            samples = list(mrag_dataset.load_scenario(perspective_type))
            assert len(samples) >= 0

            for sample in samples:
                assert isinstance(sample, Sample)
                assert sample.perspective_type == perspective_type
                assert sample.question_id is not None
                assert sample.question is not None

    def test_load_scenario_invalid_type(self, mrag_dataset):
        """Test loading with invalid scenario type."""
        with pytest.raises(ValueError):
            list(mrag_dataset.load_scenario("invalid_scenario"))

    def test_get_retrieval_corpus(self, mrag_dataset):
        """Test retrieval corpus generation."""
        corpus = mrag_dataset.get_retrieval_corpus()
        assert len(corpus) == 5
        assert all(path.endswith('.jpg') for path in corpus)
        assert all(os.path.exists(path) for path in corpus)

    def test_preprocess_batch(self, mrag_dataset):
        """Test batch preprocessing."""
        # Get a few samples
        samples = list(mrag_dataset.load_scenario('angle'))[:2]

        batch_data = mrag_dataset.preprocess_batch(samples)

        assert isinstance(batch_data, BatchData)
        assert len(batch_data.samples) == len(samples)
        assert len(batch_data.images) == len(samples)
        assert len(batch_data.questions) == len(samples)
        assert batch_data.batch_size == len(samples)

    def test_preprocess_for_clip(self, mrag_dataset):
        """Test CLIP-specific preprocessing."""
        # Create test images
        images = [Image.new('RGB', (224, 224), color=(i*50, i*50, i*50)) for i in range(3)]

        tensor = mrag_dataset.preprocess_for_clip(images)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_validate_dataset(self, mrag_dataset):
        """Test dataset validation."""
        validation_result = mrag_dataset.validate_dataset()

        assert "status" in validation_result
        assert validation_result["total_samples"] == 5
        assert validation_result["total_images"] >= 0

    def test_get_scenario_stats(self, mrag_dataset):
        """Test scenario statistics."""
        stats = mrag_dataset.get_scenario_stats()

        assert isinstance(stats, dict)
        assert len(stats) == 4  # Four perspective types
        assert all(isinstance(count, int) for count in stats.values())
        assert sum(stats.values()) == 5  # Total samples

    def test_create_dataloader(self, mrag_dataset):
        """Test PyTorch DataLoader creation."""
        dataloader = mrag_dataset.create_dataloader('angle', shuffle=False)

        assert dataloader is not None
        assert dataloader.batch_size == mrag_dataset.batch_size

        # Test loading a batch
        for batch in dataloader:
            assert 'image' in batch
            assert 'question' in batch
            assert 'question_id' in batch
            break  # Just test first batch

    def test_sample_creation_from_question(self, mrag_dataset):
        """Test sample creation from question data."""
        question_data = {
            "question_id": "test_001",
            "question": "Test question?",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "image_path": "images/image_000000.jpg",
            "category": "angle_change",
            "scenario": "angle_change"
        }

        sample = mrag_dataset._create_sample_from_question(question_data)

        assert sample is not None
        assert sample.question_id == "test_001"
        assert sample.question == "Test question?"
        assert sample.ground_truth == "A"
        assert sample.perspective_type in mrag_dataset.perspective_types

    def test_image_transform_setup(self, mrag_dataset):
        """Test image transform pipeline setup."""
        assert mrag_dataset.image_transform is not None
        assert mrag_dataset.basic_transform is not None

        # Test transform on dummy image
        img = Image.new('RGB', (100, 100), color='red')
        transformed = mrag_dataset.image_transform(img)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)


class TestImagePreprocessor:
    """Test suite for ImagePreprocessor."""

    @pytest.fixture
    def preprocessor_config(self):
        """Create preprocessing configuration."""
        return PreprocessingConfig(
            image_size=(224, 224),
            clip_normalization=True,
            enable_augmentation=False
        )

    @pytest.fixture
    def preprocessor(self, preprocessor_config):
        """Create ImagePreprocessor instance."""
        return ImagePreprocessor(preprocessor_config)

    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.config is not None
        assert preprocessor.standard_transform is not None
        assert preprocessor.basic_transform is not None

    def test_preprocess_single_image(self, preprocessor):
        """Test single image preprocessing."""
        img = Image.new('RGB', (100, 100), color='blue')
        tensor = preprocessor.preprocess_image(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_preprocess_image_with_scenario(self, preprocessor):
        """Test preprocessing with scenario-specific adjustments."""
        img = Image.new('RGB', (100, 100), color='green')

        for scenario in ['angle', 'partial', 'scope', 'occlusion']:
            tensor = preprocessor.preprocess_image(img, scenario_type=scenario)
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (3, 224, 224)

    def test_preprocess_batch_images(self, preprocessor):
        """Test batch image preprocessing."""
        images = [Image.new('RGB', (100, 100), color=(i*50, i*50, i*50)) for i in range(3)]
        tensor_batch = preprocessor.preprocess_batch(images)

        assert isinstance(tensor_batch, torch.Tensor)
        assert tensor_batch.shape == (3, 3, 224, 224)

    def test_preprocess_invalid_image(self, preprocessor):
        """Test preprocessing with invalid image path."""
        tensor = preprocessor.preprocess_image("nonexistent_file.jpg")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)


class TestMemoryAwareDataLoader:
    """Test suite for MemoryAwareDataLoader."""

    @pytest.fixture
    def streaming_config(self):
        """Create streaming configuration."""
        return StreamingConfig(
            chunk_size=5,
            prefetch_size=1,
            enable_caching=True,
            cache_size_mb=50,
            auto_batch_sizing=False  # Disable for testing
        )

    @pytest.fixture
    def preprocessing_config(self):
        """Create preprocessing configuration."""
        return PreprocessingConfig(
            image_size=(224, 224),
            enable_augmentation=False
        )

    @pytest.fixture
    def memory_aware_loader(self, temp_dataset_dir, streaming_config, preprocessing_config):
        """Create MemoryAwareDataLoader instance."""
        dataset = MRAGDataset(temp_dataset_dir, batch_size=2)
        return MemoryAwareDataLoader(dataset, streaming_config, preprocessing_config)

    def test_loader_initialization(self, memory_aware_loader):
        """Test data loader initialization."""
        assert memory_aware_loader.dataset is not None
        assert memory_aware_loader.preprocessor is not None
        assert memory_aware_loader.batch_processor is not None

    def test_create_streaming_loader(self, memory_aware_loader):
        """Test streaming loader creation."""
        # Test with a small dataset to avoid long processing
        batch_count = 0
        for batch in memory_aware_loader.create_streaming_loader('angle', batch_size=1):
            assert 'images' in batch
            assert isinstance(batch['images'], torch.Tensor)
            batch_count += 1
            if batch_count >= 2:  # Limit test to 2 batches
                break

        assert batch_count > 0

    def test_get_performance_stats(self, memory_aware_loader):
        """Test performance statistics collection."""
        stats = memory_aware_loader.get_performance_stats()
        assert isinstance(stats, dict)
        assert 'samples_processed' in stats
        assert 'batches_created' in stats
        assert 'memory_stats' in stats

    def test_validate_memory_efficiency(self, memory_aware_loader):
        """Test memory efficiency validation."""
        result = memory_aware_loader.validate_memory_efficiency('angle', test_batches=2)
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'batches_processed' in result


class TestDatasetValidator:
    """Test suite for DatasetValidator."""

    @pytest.fixture
    def validator(self):
        """Create DatasetValidator instance."""
        return DatasetValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.memory_manager is not None
        assert len(validator.supported_formats) > 0

    def test_validate_sample_batch(self, validator, temp_dataset_dir):
        """Test sample batch validation."""
        dataset = MRAGDataset(temp_dataset_dir, batch_size=2)
        result = validator.validate_sample_batch(dataset, 'angle', batch_size=2)

        assert isinstance(result, dict)
        assert 'scenario' in result
        assert 'samples_tested' in result
        assert result['scenario'] == 'angle'

    @patch('PIL.Image.open')
    def test_validate_dataset_with_mock(self, mock_image_open, validator, temp_dataset_dir):
        """Test dataset validation with mocked image operations."""
        # Mock image opening to avoid actual file operations
        mock_img = Mock()
        mock_img.size = (224, 224)
        mock_img.format = 'JPEG'
        mock_img.mode = 'RGB'
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value.__enter__.return_value = mock_img

        dataset = MRAGDataset(temp_dataset_dir, batch_size=2)
        result = validator.validate_dataset(dataset, check_images=True, sample_limit=3)

        assert isinstance(result, ValidationResult)
        assert result.status in ['success', 'warning', 'error']
        assert result.total_samples >= 0

    def test_generate_validation_report(self, validator, temp_dataset_dir):
        """Test validation report generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_path = f.name

        try:
            result = ValidationResult(
                status="success",
                total_samples=5,
                valid_samples=5,
                scenario_distribution={'angle': 2, 'partial': 1, 'scope': 1, 'occlusion': 1}
            )

            validator.generate_validation_report(result, report_path)
            assert os.path.exists(report_path)

            # Check report content
            with open(report_path, 'r') as f:
                report_data = json.load(f)

            assert 'validation_summary' in report_data
            assert 'detailed_results' in report_data
            assert 'recommendations' in report_data

        finally:
            if os.path.exists(report_path):
                os.unlink(report_path)


# Integration tests
class TestDatasetIntegration:
    """Integration tests for dataset components."""

    def test_end_to_end_processing(self, temp_dataset_dir):
        """Test complete end-to-end dataset processing."""
        # Initialize components
        dataset = MRAGDataset(temp_dataset_dir, batch_size=2)
        validator = DatasetValidator()

        # Validate dataset
        validation_result = validator.validate_dataset(dataset, sample_limit=3)
        assert validation_result.status in ['success', 'warning']

        # Test scenario loading
        samples = list(dataset.load_scenario('angle'))
        assert len(samples) >= 0

        if samples:
            # Test preprocessing
            batch_data = dataset.preprocess_batch(samples[:2])
            assert isinstance(batch_data, BatchData)

            # Test CLIP preprocessing
            if batch_data.images:
                clip_tensors = dataset.preprocess_for_clip([img for img in batch_data.images if img is not None])
                assert isinstance(clip_tensors, torch.Tensor)

    def test_memory_management_integration(self, temp_dataset_dir):
        """Test memory management during dataset operations."""
        memory_manager = MemoryManager(memory_limit_gb=2.0)  # Low limit for testing
        dataset = MRAGDataset(temp_dataset_dir, batch_size=1)

        # Test with memory monitoring
        with memory_manager.memory_guard("test_operation"):
            samples = list(dataset.load_scenario('angle'))
            if samples:
                batch_data = dataset.preprocess_batch(samples[:1])
                assert batch_data is not None

        # Check that operation completed without memory errors
        assert True  # If we reach here, memory management worked


if __name__ == "__main__":
    # Run specific test if script is executed directly
    pytest.main([__file__, "-v"])