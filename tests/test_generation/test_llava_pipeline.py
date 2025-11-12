"""
Unit Tests for LLaVA Generation Pipeline

Comprehensive tests for LLaVA-1.5-7B integration with 4-bit quantization.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import tempfile
import os
from typing import List

from src.generation.llava_pipeline import LLaVAGenerationPipeline
from src.generation.interface import GenerationConfig, MultimodalContext, GenerationResult


class TestLLaVAGenerationPipeline:
    """Test suite for LLaVA generation pipeline."""

    @pytest.fixture
    def mock_config(self) -> GenerationConfig:
        """Create mock generation configuration."""
        return GenerationConfig(
            model_name="llava-hf/llava-1.5-7b-hf",
            max_length=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            quantization="4bit",
            max_memory_gb=5.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="float16"
        )

    @pytest.fixture
    def sample_images(self) -> List[Image.Image]:
        """Create sample images for testing."""
        images = []
        for i in range(3):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(img_array, 'RGB')
            images.append(image)
        return images

    @pytest.fixture
    def sample_context(self, sample_images) -> MultimodalContext:
        """Create sample multimodal context."""
        return MultimodalContext(
            question="What do you see in this medical image?",
            images=sample_images[:2],  # Use first 2 images
            image_paths=["/path/to/image1.jpg", "/path/to/image2.jpg"],
            context_metadata={"question_id": "test_001"}
        )

    def test_pipeline_initialization(self, mock_config):
        """Test pipeline initialization."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        assert pipeline.config == mock_config
        assert pipeline.model is None
        assert pipeline.processor is None
        assert not pipeline.model_loaded
        assert pipeline.memory_manager is not None

    def test_quantization_config_creation(self, mock_config):
        """Test quantization configuration creation."""
        pipeline = LLaVAGenerationPipeline(mock_config)
        quant_config = pipeline._create_quantization_config()

        assert quant_config.load_in_4bit is True
        assert quant_config.bnb_4bit_compute_dtype == torch.float16
        assert quant_config.bnb_4bit_use_double_quant is True
        assert quant_config.bnb_4bit_quant_type == "nf4"

    @patch('src.generation.llava_pipeline.LlavaNextProcessor')
    @patch('src.generation.llava_pipeline.LlavaNextForConditionalGeneration')
    def test_model_loading(self, mock_model_class, mock_processor_class, mock_config):
        """Test model loading with quantization."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.tokenizer.pad_token_id = 0
        mock_processor.tokenizer.eos_token_id = 2
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        pipeline = LLaVAGenerationPipeline(mock_config)

        # Test loading
        pipeline.load_model()

        # Verify calls
        mock_processor_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

        # Verify state
        assert pipeline.model_loaded is True
        assert pipeline.model == mock_model
        assert pipeline.processor == mock_processor

    def test_image_preparation(self, mock_config, sample_images):
        """Test image preparation and validation."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        # Test with valid images
        prepared_images = pipeline._prepare_images(sample_images)
        assert len(prepared_images) == len(sample_images)

        # Test with None image
        images_with_none = sample_images + [None]
        prepared_images = pipeline._prepare_images(images_with_none)
        assert len(prepared_images) == len(sample_images)  # None should be filtered out

        # Test with very small image
        small_image = Image.new('RGB', (10, 10), color='red')
        prepared_images = pipeline._prepare_images([small_image])
        assert len(prepared_images) == 0  # Too small, should be filtered

        # Test with large image (should be resized)
        large_image = Image.new('RGB', (1024, 1024), color='blue')
        prepared_images = pipeline._prepare_images([large_image])
        assert len(prepared_images) == 1
        assert max(prepared_images[0].size) <= 512  # Should be resized

    def test_prompt_construction(self, mock_config, sample_context):
        """Test multimodal prompt construction."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        prompt = pipeline.construct_prompt(sample_context)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "medical" in prompt.lower()
        assert sample_context.question in prompt
        assert "Answer:" in prompt

        # Test with different number of images
        context_single_image = MultimodalContext(
            question="What is this?",
            images=[sample_context.images[0]]
        )

        prompt_single = pipeline.construct_prompt(context_single_image)
        assert "medical image provided" in prompt_single

        # Test with no images
        context_no_images = MultimodalContext(
            question="What is this?",
            images=[]
        )

        prompt_no_images = pipeline.construct_prompt(context_no_images)
        assert "medical image provided" not in prompt_no_images

    def test_response_post_processing(self, mock_config):
        """Test response post-processing."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        # Test basic cleaning
        response = "  this is a test response  "
        cleaned = pipeline._post_process_response(response)
        assert cleaned == "This is a test response."

        # Test capitalization
        response = "lowercase start"
        cleaned = pipeline._post_process_response(response)
        assert cleaned.startswith("L")

        # Test punctuation
        response = "No punctuation"
        cleaned = pipeline._post_process_response(response)
        assert cleaned.endswith(".")

        # Test duplicate line removal
        response = "Line 1\nLine 1\nLine 2"
        cleaned = pipeline._post_process_response(response)
        assert cleaned.count("Line 1") == 1

    def test_confidence_score_calculation(self, mock_config, sample_context):
        """Test confidence score calculation."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        # Test normal response
        response = "This medical image shows clear anatomical structures."
        confidence = pipeline._calculate_confidence_score(response, sample_context)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident

        # Test error response
        response = "I cannot determine what this shows."
        confidence = pipeline._calculate_confidence_score(response, sample_context)
        assert confidence <= 0.3  # Should have low confidence

        # Test short response
        response = "Yes."
        confidence = pipeline._calculate_confidence_score(response, sample_context)
        assert confidence < 0.7  # Should be less confident for short responses

    @patch('src.generation.llava_pipeline.LlavaNextProcessor')
    @patch('src.generation.llava_pipeline.LlavaNextForConditionalGeneration')
    def test_generation_with_mocked_model(self, mock_model_class, mock_processor_class, mock_config, sample_context):
        """Test answer generation with mocked model."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.tokenizer.pad_token_id = 0
        mock_processor.tokenizer.eos_token_id = 2
        mock_processor.tokenizer.decode.return_value = "This is a test medical response."
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }

        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        pipeline = LLaVAGenerationPipeline(mock_config)
        pipeline.load_model()

        # Test generation
        result = pipeline.generate_answer(sample_context)

        assert isinstance(result, GenerationResult)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.generation_time > 0
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.memory_usage, dict)

    def test_memory_management(self, mock_config):
        """Test memory management functions."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        # Test memory usage reporting
        memory_usage = pipeline.get_memory_usage()
        assert isinstance(memory_usage, dict)
        assert "gpu_allocated_gb" in memory_usage

        # Test memory constraint validation
        is_valid = pipeline.validate_memory_constraints()
        assert isinstance(is_valid, bool)

        # Test memory clearing (should not raise errors)
        pipeline.clear_memory()

    @patch('src.generation.llava_pipeline.LlavaNextProcessor')
    @patch('src.generation.llava_pipeline.LlavaNextForConditionalGeneration')
    def test_model_unloading(self, mock_model_class, mock_processor_class, mock_config):
        """Test model unloading and cleanup."""
        # Setup mocks
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model

        pipeline = LLaVAGenerationPipeline(mock_config)
        pipeline.load_model()

        assert pipeline.model_loaded is True

        # Test unloading
        pipeline.unload_model()

        assert pipeline.model_loaded is False
        assert pipeline.model is None
        assert pipeline.processor is None

    def test_generation_with_no_images(self, mock_config):
        """Test generation when no valid images are provided."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        context = MultimodalContext(
            question="What do you see?",
            images=[],  # No images
            image_paths=[]
        )

        # Should not crash and return appropriate response
        result = pipeline.generate_answer(context)
        assert isinstance(result, GenerationResult)
        assert "cannot provide an answer" in result.answer.lower()
        assert result.confidence_score == 0.0

    def test_generation_with_invalid_images(self, mock_config):
        """Test generation with invalid/corrupted images."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        # Create invalid images
        invalid_images = [None, "not an image", 123]

        context = MultimodalContext(
            question="What do you see?",
            images=invalid_images,
            image_paths=[]
        )

        # Should handle gracefully
        result = pipeline.generate_answer(context)
        assert isinstance(result, GenerationResult)
        assert result.confidence_score == 0.0

    @pytest.mark.parametrize("max_memory_gb", [1.0, 5.0, 10.0])
    def test_different_memory_limits(self, max_memory_gb):
        """Test pipeline with different memory limits."""
        config = GenerationConfig(
            model_name="llava-hf/llava-1.5-7b-hf",
            max_memory_gb=max_memory_gb
        )

        pipeline = LLaVAGenerationPipeline(config)
        assert pipeline.config.max_memory_gb == max_memory_gb
        assert pipeline.memory_manager.memory_limit_gb == max_memory_gb + 2.0

    def test_error_handling_in_generation(self, mock_config, sample_context):
        """Test error handling during generation."""
        pipeline = LLaVAGenerationPipeline(mock_config)

        # Force an error by not loading the model
        pipeline.model_loaded = False
        pipeline.model = None

        # Generation should handle the error gracefully
        with patch.object(pipeline, 'load_model', side_effect=Exception("Mock error")):
            result = pipeline.generate_answer(sample_context)

            assert isinstance(result, GenerationResult)
            assert "error" in result.answer.lower()
            assert result.confidence_score == 0.0
            assert "error" in result.metadata


# Integration-style tests
class TestLLaVAIntegration:
    """Integration tests for LLaVA pipeline."""

    @pytest.fixture
    def temp_image_file(self):
        """Create temporary image file."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # Create a simple test image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name, 'JPEG')
            yield f.name
        os.unlink(f.name)

    def test_pipeline_with_file_images(self, temp_image_file):
        """Test pipeline with actual image files."""
        config = GenerationConfig(
            model_name="llava-hf/llava-1.5-7b-hf",
            max_memory_gb=2.0
        )

        pipeline = LLaVAGenerationPipeline(config)

        # Load image from file
        image = Image.open(temp_image_file)
        context = MultimodalContext(
            question="What color is this image?",
            images=[image],
            image_paths=[temp_image_file]
        )

        # Test prompt construction (doesn't require model loading)
        prompt = pipeline.construct_prompt(context)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Test image preparation
        prepared_images = pipeline._prepare_images([image])
        assert len(prepared_images) == 1
        assert isinstance(prepared_images[0], Image.Image)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])