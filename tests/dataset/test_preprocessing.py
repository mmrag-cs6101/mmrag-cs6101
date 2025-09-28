"""
Unit Tests for Image Preprocessing Pipeline

Test suite for image preprocessing and augmentation functionality.
"""

import pytest
import torch
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import Mock, patch
import tempfile
import os

import sys
sys.path.append('/mnt/d/dev/mmrag-cs6101/src')

from dataset.preprocessing import (
    ImagePreprocessor,
    PreprocessingConfig,
    BatchDataProcessor
)
from utils.memory_manager import MemoryManager


class TestPreprocessingConfig:
    """Test preprocessing configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        assert config.image_size == (224, 224)
        assert config.clip_normalization is True
        assert config.enable_augmentation is False
        assert config.max_batch_size == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = PreprocessingConfig(
            image_size=(512, 512),
            clip_normalization=False,
            enable_augmentation=True,
            max_batch_size=16
        )
        assert config.image_size == (512, 512)
        assert config.clip_normalization is False
        assert config.enable_augmentation is True
        assert config.max_batch_size == 16


class TestImagePreprocessor:
    """Test image preprocessing functionality."""

    @pytest.fixture
    def basic_config(self):
        """Basic preprocessing configuration."""
        return PreprocessingConfig(
            image_size=(224, 224),
            clip_normalization=True,
            enable_augmentation=False
        )

    @pytest.fixture
    def augmentation_config(self):
        """Configuration with augmentation enabled."""
        return PreprocessingConfig(
            image_size=(224, 224),
            clip_normalization=True,
            enable_augmentation=True,
            augmentation_probability=0.5
        )

    @pytest.fixture
    def preprocessor(self, basic_config):
        """Create basic preprocessor."""
        return ImagePreprocessor(basic_config)

    @pytest.fixture
    def augmented_preprocessor(self, augmentation_config):
        """Create preprocessor with augmentation."""
        return ImagePreprocessor(augmentation_config)

    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.config is not None
        assert preprocessor.memory_manager is not None
        assert preprocessor.standard_transform is not None
        assert preprocessor.basic_transform is not None
        assert preprocessor.augmented_transform is not None

    def test_transform_setup_clip_normalization(self, preprocessor):
        """Test CLIP normalization setup."""
        # Check that CLIP normalization values are used
        transform = preprocessor.standard_transform

        # Find the normalization transform
        normalize_transform = None
        for t in transform.transforms:
            if hasattr(t, 'mean') and hasattr(t, 'std'):
                normalize_transform = t
                break

        assert normalize_transform is not None
        expected_mean = [0.48145466, 0.4578275, 0.40821073]
        expected_std = [0.26862954, 0.26130258, 0.27577711]

        assert np.allclose(normalize_transform.mean, expected_mean, atol=1e-6)
        assert np.allclose(normalize_transform.std, expected_std, atol=1e-6)

    def test_preprocess_single_image_pil(self, preprocessor):
        """Test preprocessing a PIL image."""
        # Create test image
        img = Image.new('RGB', (100, 150), color='red')

        result = preprocessor.preprocess_image(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_preprocess_single_image_path(self, preprocessor):
        """Test preprocessing from image path."""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (100, 150), color='blue')
            img.save(f.name)
            temp_path = f.name

        try:
            result = preprocessor.preprocess_image(temp_path)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)

        finally:
            os.unlink(temp_path)

    def test_preprocess_invalid_path(self, preprocessor):
        """Test preprocessing with invalid image path."""
        result = preprocessor.preprocess_image("nonexistent_file.jpg")

        # Should return placeholder image tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_preprocess_with_augmentation(self, augmented_preprocessor):
        """Test preprocessing with augmentation."""
        img = Image.new('RGB', (100, 150), color='green')

        # Test with augmentation enabled
        result = augmented_preprocessor.preprocess_image(img, apply_augmentation=True)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_scenario_specific_preprocessing(self, preprocessor):
        """Test scenario-specific preprocessing."""
        img = Image.new('RGB', (200, 200), color='purple')

        scenarios = ['angle', 'partial', 'scope', 'occlusion']

        for scenario in scenarios:
            result = preprocessor.preprocess_image(img, scenario_type=scenario)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)

    def test_batch_preprocessing(self, preprocessor):
        """Test batch image preprocessing."""
        # Create test images
        images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (150, 150), color='green'),
            Image.new('RGB', (200, 200), color='blue')
        ]

        result = preprocessor.preprocess_batch(images)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3, 224, 224)

    def test_batch_preprocessing_with_scenarios(self, preprocessor):
        """Test batch preprocessing with scenario types."""
        images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (150, 150), color='green')
        ]
        scenario_types = ['angle', 'partial']

        result = preprocessor.preprocess_batch(images, scenario_types=scenario_types)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3, 224, 224)

    def test_empty_batch(self, preprocessor):
        """Test preprocessing empty batch."""
        result = preprocessor.preprocess_batch([])

        assert isinstance(result, torch.Tensor)
        assert result.shape == (0, 3, 224, 224)

    def test_contrast_enhancement(self, preprocessor):
        """Test contrast enhancement method."""
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))

        enhanced = preprocessor._enhance_contrast(img, factor=1.2)

        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == img.size

    def test_sharpness_enhancement(self, preprocessor):
        """Test sharpness enhancement method."""
        img = Image.new('RGB', (100, 100), color='gray')

        enhanced = preprocessor._enhance_sharpness(img, factor=1.1)

        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == img.size

    def test_brightness_normalization(self, preprocessor):
        """Test brightness normalization method."""
        # Create image with specific brightness
        img_array = np.full((100, 100, 3), 200, dtype=np.uint8)  # Bright image
        img = Image.fromarray(img_array)

        normalized = preprocessor._normalize_brightness(img)

        assert isinstance(normalized, Image.Image)
        assert normalized.size == img.size

        # Check that brightness changed
        normalized_array = np.array(normalized)
        original_brightness = np.mean(img_array)
        new_brightness = np.mean(normalized_array)

        assert new_brightness != original_brightness

    def test_placeholder_image_creation(self, preprocessor):
        """Test placeholder image creation."""
        placeholder = preprocessor._create_placeholder_image()

        assert isinstance(placeholder, Image.Image)
        assert placeholder.size == preprocessor.config.image_size
        assert placeholder.mode == 'RGB'

    def test_get_transform_for_clip(self, preprocessor):
        """Test getting CLIP transform."""
        transform = preprocessor.get_transform_for_clip()

        assert transform is not None
        assert transform == preprocessor.standard_transform

    def test_get_basic_transform(self, preprocessor):
        """Test getting basic transform."""
        transform = preprocessor.get_basic_transform()

        assert transform is not None
        assert transform == preprocessor.basic_transform

    def test_memory_management_integration(self):
        """Test preprocessor with memory management."""
        memory_manager = MemoryManager(memory_limit_gb=1.0)
        config = PreprocessingConfig(max_batch_size=2)

        preprocessor = ImagePreprocessor(config, memory_manager)

        # Test that memory manager is used
        assert preprocessor.memory_manager == memory_manager

    @patch('src.utils.memory_manager.MemoryManager.check_memory_availability')
    def test_memory_aware_batch_processing(self, mock_memory_check, preprocessor):
        """Test memory-aware batch processing."""
        # Mock memory availability
        mock_memory_check.return_value = False

        images = [Image.new('RGB', (100, 100), color='red') for _ in range(10)]

        # Should still process despite memory constraints
        result = preprocessor.preprocess_batch(images)

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 10  # All images processed


class TestBatchDataProcessor:
    """Test batch data processing functionality."""

    @pytest.fixture
    def config(self):
        """Create preprocessing configuration."""
        return PreprocessingConfig(image_size=(224, 224))

    @pytest.fixture
    def preprocessor(self, config):
        """Create preprocessor."""
        return ImagePreprocessor(config)

    @pytest.fixture
    def batch_processor(self, preprocessor):
        """Create batch processor."""
        return BatchDataProcessor(preprocessor)

    def test_batch_processor_initialization(self, batch_processor):
        """Test batch processor initialization."""
        assert batch_processor.preprocessor is not None
        assert batch_processor.memory_manager is not None

    def test_process_batch_with_images(self, batch_processor):
        """Test processing batch with images."""
        batch_data = {
            'images': [
                Image.new('RGB', (100, 100), color='red'),
                Image.new('RGB', (150, 150), color='blue')
            ],
            'questions': ['Question 1?', 'Question 2?'],
            'labels': ['A', 'B']
        }

        result = batch_processor.process_batch(batch_data, target_device='cpu')

        assert 'images' in result
        assert isinstance(result['images'], torch.Tensor)
        assert result['images'].shape == (2, 3, 224, 224)
        assert 'questions' in result
        assert 'labels' in result

    def test_process_batch_without_images(self, batch_processor):
        """Test processing batch without images."""
        batch_data = {
            'questions': ['Question 1?', 'Question 2?'],
            'labels': ['A', 'B']
        }

        result = batch_processor.process_batch(batch_data)

        assert 'questions' in result
        assert 'labels' in result
        assert 'images' not in result

    def test_process_batch_with_tensors(self, batch_processor):
        """Test processing batch with existing tensors."""
        tensor_data = torch.randn(2, 10)

        batch_data = {
            'features': tensor_data,
            'questions': ['Question 1?', 'Question 2?']
        }

        result = batch_processor.process_batch(batch_data, target_device='cpu')

        assert 'features' in result
        assert isinstance(result['features'], torch.Tensor)
        assert torch.equal(result['features'], tensor_data)

    def test_get_optimal_batch_size(self, batch_processor):
        """Test optimal batch size calculation."""
        optimal_size = batch_processor.get_optimal_batch_size(
            base_batch_size=16,
            image_size=(224, 224)
        )

        assert isinstance(optimal_size, int)
        assert optimal_size > 0
        assert optimal_size <= 16  # Should not exceed base size

    @patch('torch.cuda.is_available')
    def test_cuda_device_handling(self, mock_cuda_available, batch_processor):
        """Test CUDA device handling."""
        mock_cuda_available.return_value = False

        batch_data = {
            'images': [Image.new('RGB', (100, 100), color='red')],
            'questions': ['Question?']
        }

        result = batch_processor.process_batch(batch_data, target_device='cuda')

        # Should handle gracefully when CUDA not available
        assert 'images' in result
        assert isinstance(result['images'], torch.Tensor)


class TestImageQuality:
    """Test image quality and preprocessing correctness."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor for quality tests."""
        config = PreprocessingConfig(image_size=(224, 224))
        return ImagePreprocessor(config)

    def test_image_aspect_ratio_handling(self, preprocessor):
        """Test handling of different aspect ratios."""
        # Test with different aspect ratios
        test_cases = [
            (100, 200),  # Tall
            (200, 100),  # Wide
            (100, 100),  # Square
            (300, 150),  # 2:1 ratio
        ]

        for width, height in test_cases:
            img = Image.new('RGB', (width, height), color='red')
            result = preprocessor.preprocess_image(img)

            # Should always output 224x224
            assert result.shape == (3, 224, 224)

    def test_color_mode_handling(self, preprocessor):
        """Test handling of different color modes."""
        # Test different color modes
        modes = ['RGB', 'L', 'RGBA', 'P']

        for mode in modes:
            if mode == 'L':
                img = Image.new(mode, (100, 100), color=128)
            elif mode == 'P':
                img = Image.new('RGB', (100, 100), color='red').convert('P')
            else:
                img = Image.new(mode, (100, 100), color='red')

            result = preprocessor.preprocess_image(img)

            # Should always output RGB tensor
            assert result.shape == (3, 224, 224)

    def test_normalization_range(self, preprocessor):
        """Test that normalization produces expected value ranges."""
        # Create test image with known values
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)  # Mid-gray
        img = Image.fromarray(img_array)

        result = preprocessor.preprocess_image(img)

        # Values should be normalized (typically in range roughly -2 to 2)
        assert result.min() >= -3.0
        assert result.max() <= 3.0

        # Should not be in 0-1 range (which would indicate missing normalization)
        assert not (result.min() >= 0.0 and result.max() <= 1.0)

    def test_batch_consistency(self, preprocessor):
        """Test that batch processing gives same results as individual processing."""
        images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (150, 150), color='green')
        ]

        # Process individually
        individual_results = [preprocessor.preprocess_image(img) for img in images]
        individual_batch = torch.stack(individual_results)

        # Process as batch
        batch_result = preprocessor.preprocess_batch(images)

        # Results should be identical
        assert torch.allclose(individual_batch, batch_result, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])