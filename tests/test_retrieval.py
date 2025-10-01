"""
Unit Tests for CLIP Retrieval System

Comprehensive test suite for Sprint 3 CLIP-based image retrieval implementation.
Tests all components: CLIPRetriever, EmbeddingProcessor, and RetrievalFactory.
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import torch

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.retrieval import (
    CLIPRetriever, EmbeddingProcessor, RetrievalFactory,
    RetrievalConfig, RetrievalResult
)
from src.config import MRAGConfig
from src.utils.memory_manager import MemoryManager


class TestRetrievalConfig(unittest.TestCase):
    """Test RetrievalConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetrievalConfig()

        self.assertEqual(config.model_name, "openai/clip-vit-base-patch32")
        self.assertEqual(config.embedding_dim, 512)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.device, "cuda")

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetrievalConfig(
            model_name="custom/clip-model",
            embedding_dim=768,
            top_k=10,
            batch_size=32,
            device="cpu"
        )

        self.assertEqual(config.model_name, "custom/clip-model")
        self.assertEqual(config.embedding_dim, 768)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.device, "cpu")


class TestRetrievalResult(unittest.TestCase):
    """Test RetrievalResult dataclass."""

    def test_result_creation(self):
        """Test creating retrieval results."""
        result = RetrievalResult(
            image_path="test.jpg",
            similarity_score=0.85
        )

        self.assertEqual(result.image_path, "test.jpg")
        self.assertEqual(result.similarity_score, 0.85)
        self.assertIsNone(result.embedding)
        self.assertEqual(result.metadata, {})

    def test_result_with_metadata(self):
        """Test result with metadata."""
        metadata = {"category": "medical", "index": 42}
        result = RetrievalResult(
            image_path="test.jpg",
            similarity_score=0.85,
            metadata=metadata
        )

        self.assertEqual(result.metadata, metadata)


class TestCLIPRetriever(unittest.TestCase):
    """Test CLIPRetriever implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RetrievalConfig(
            model_name="openai/clip-vit-base-patch32",
            embedding_dim=512,
            batch_size=2,
            device="cpu"  # Use CPU for tests
        )
        self.retriever = CLIPRetriever(self.config)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'retriever'):
            self.retriever.clear_memory()

    @patch('src.retrieval.clip_retriever.CLIPModel')
    @patch('src.retrieval.clip_retriever.CLIPProcessor')
    def test_model_loading(self, mock_processor, mock_model):
        """Test CLIP model loading."""
        # Mock the model and processor
        mock_model_instance = Mock()
        mock_processor_instance = Mock()

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance

        # Test model loading
        self.retriever._load_model()

        # Verify model was loaded
        self.assertIsNotNone(self.retriever.model)
        self.assertIsNotNone(self.retriever.processor)

        # Verify correct methods were called
        mock_model.from_pretrained.assert_called_once()
        mock_processor.from_pretrained.assert_called_once()

    def test_encode_images_empty_list(self):
        """Test encoding empty image list."""
        embeddings = self.retriever.encode_images([])

        self.assertEqual(embeddings.shape, (0, 512))

    def test_encode_text_empty_list(self):
        """Test encoding empty text list."""
        embeddings = self.retriever.encode_text([])

        self.assertEqual(embeddings.shape, (0, 512))

    @patch('src.retrieval.clip_retriever.CLIPModel')
    @patch('src.retrieval.clip_retriever.CLIPProcessor')
    def test_encode_images_with_mock(self, mock_processor, mock_model):
        """Test image encoding with mocked model."""
        # Setup mocks
        mock_processor_instance = Mock()
        mock_model_instance = Mock()

        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock processor output
        mock_processor_instance.return_value = {
            "pixel_values": torch.randn(2, 3, 224, 224)
        }

        # Mock model output
        mock_features = torch.randn(2, 512)
        mock_model_instance.get_image_features.return_value = mock_features

        # Create test images
        test_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue')
        ]

        # Test encoding
        embeddings = self.retriever.encode_images(test_images)

        # Verify shape
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 512)

    def test_build_index_empty_embeddings(self):
        """Test building index with empty embeddings."""
        empty_embeddings = np.empty((0, 512))
        empty_paths = []

        with self.assertRaises(Exception):
            self.retriever.build_index(empty_embeddings, empty_paths)

    def test_build_index_mismatch(self):
        """Test building index with mismatched embeddings and paths."""
        embeddings = np.random.randn(5, 512).astype(np.float32)
        paths = ["path1.jpg", "path2.jpg"]  # Wrong count

        with self.assertRaises(Exception):
            self.retriever.build_index(embeddings, paths)

    def test_build_index_success(self):
        """Test successful index building."""
        # Create test embeddings
        num_samples = 10
        embeddings = np.random.randn(num_samples, 512).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        paths = [f"image_{i}.jpg" for i in range(num_samples)]

        # Build index
        self.retriever.build_index(embeddings, paths)

        # Verify index was created
        self.assertIsNotNone(self.retriever.index)
        self.assertEqual(self.retriever.index.ntotal, num_samples)
        self.assertEqual(len(self.retriever.image_paths), num_samples)

    def test_retrieve_without_index(self):
        """Test retrieval without building index first."""
        with self.assertRaises(Exception):
            self.retriever.retrieve_similar("test query")

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        memory_stats = self.retriever.get_memory_usage()

        self.assertIn("gpu_allocated", memory_stats)
        self.assertIn("gpu_reserved", memory_stats)
        self.assertIn("gpu_total", memory_stats)

    def test_warmup(self):
        """Test model warmup functionality."""
        with patch.object(self.retriever, 'encode_images') as mock_encode_images, \
             patch.object(self.retriever, 'encode_text') as mock_encode_text:

            mock_encode_images.return_value = np.zeros((4, 512))
            mock_encode_text.return_value = np.zeros((2, 512))

            stats = self.retriever.warmup(num_images=4, num_texts=2)

            # Verify warmup was called
            mock_encode_images.assert_called_once()
            mock_encode_text.assert_called_once()

            # Verify stats returned
            self.assertIn("image_encoding_time", stats)
            self.assertIn("text_encoding_time", stats)


class TestEmbeddingProcessor(unittest.TestCase):
    """Test EmbeddingProcessor implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RetrievalConfig(device="cpu", batch_size=2)
        self.retriever = CLIPRetriever(self.config)
        self.temp_dir = tempfile.mkdtemp()
        self.processor = EmbeddingProcessor(
            self.retriever,
            cache_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if hasattr(self, 'retriever'):
            self.retriever.clear_memory()

    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor.retriever)
        self.assertIsNotNone(self.processor.memory_manager)
        self.assertEqual(str(self.processor.cache_dir), self.temp_dir)

    def test_empty_image_corpus(self):
        """Test processing empty image corpus."""
        embeddings, paths = self.processor.process_image_corpus([])

        self.assertEqual(embeddings.shape, (0, 512))
        self.assertEqual(len(paths), 0)

    def test_nonexistent_images(self):
        """Test processing non-existent image paths."""
        fake_paths = ["nonexistent1.jpg", "nonexistent2.jpg"]

        embeddings, paths = self.processor.process_image_corpus(fake_paths)

        # Should handle gracefully
        self.assertEqual(embeddings.shape[0], 0)
        self.assertEqual(len(paths), 0)

    def test_cache_operations(self):
        """Test cache save and load operations."""
        # Add some fake processed paths
        self.processor.processed_paths.add("test1.jpg")
        self.processor.failed_paths.add("test2.jpg")

        # Save cache
        self.processor._save_cache()

        # Verify cache files exist
        self.assertTrue((Path(self.temp_dir) / "processed_paths.json").exists())
        self.assertTrue((Path(self.temp_dir) / "failed_paths.json").exists())

        # Clear and reload
        self.processor.processed_paths.clear()
        self.processor.failed_paths.clear()

        self.processor._load_cache()

        # Verify data was restored
        self.assertIn("test1.jpg", self.processor.processed_paths)
        self.assertIn("test2.jpg", self.processor.failed_paths)

    def test_clear_cache(self):
        """Test cache clearing."""
        # Add some data
        self.processor.processed_paths.add("test.jpg")
        self.processor._save_cache()

        # Clear cache
        self.processor.clear_cache()

        # Verify everything is cleared
        self.assertEqual(len(self.processor.processed_paths), 0)
        self.assertFalse((Path(self.temp_dir) / "processed_paths.json").exists())

    def test_validate_embeddings(self):
        """Test embedding validation."""
        # Create valid embeddings
        embeddings = np.random.randn(5, 512).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        paths = [f"test_{i}.jpg" for i in range(5)]

        validation = self.processor.validate_embeddings(embeddings, paths)

        self.assertEqual(validation["status"], "success")
        self.assertEqual(validation["total_embeddings"], 5)
        self.assertEqual(validation["embedding_dim"], 512)
        self.assertEqual(len(validation["errors"]), 0)

    def test_validate_embeddings_with_errors(self):
        """Test embedding validation with errors."""
        # Create embeddings with wrong dimension
        embeddings = np.random.randn(3, 256).astype(np.float32)  # Wrong dimension
        paths = [f"test_{i}.jpg" for i in range(3)]

        validation = self.processor.validate_embeddings(embeddings, paths)

        self.assertNotEqual(validation["status"], "success")
        self.assertGreater(len(validation["errors"]), 0)

    def test_processing_stats(self):
        """Test processing statistics."""
        stats = self.processor.get_processing_stats()

        self.assertIn("processed_images", stats)
        self.assertIn("failed_images", stats)
        self.assertIn("cached_embeddings", stats)
        self.assertIn("performance_stats", stats)


class TestRetrievalFactory(unittest.TestCase):
    """Test RetrievalFactory implementation."""

    def test_create_clip_retriever_default(self):
        """Test creating CLIP retriever with defaults."""
        retriever = RetrievalFactory.create_clip_retriever()

        self.assertIsInstance(retriever, CLIPRetriever)
        self.assertEqual(retriever.config.model_name, "openai/clip-vit-base-patch32")
        self.assertEqual(retriever.config.embedding_dim, 512)

    def test_create_clip_retriever_with_config(self):
        """Test creating CLIP retriever with custom config."""
        config = RetrievalConfig(
            model_name="custom/model",
            embedding_dim=768,
            top_k=10
        )

        retriever = RetrievalFactory.create_clip_retriever(config)

        self.assertEqual(retriever.config.model_name, "custom/model")
        self.assertEqual(retriever.config.embedding_dim, 768)
        self.assertEqual(retriever.config.top_k, 10)

    def test_create_embedding_processor(self):
        """Test creating embedding processor."""
        config = RetrievalConfig(device="cpu")
        retriever = CLIPRetriever(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RetrievalFactory.create_embedding_processor(
                retriever, cache_dir=temp_dir
            )

            self.assertIsInstance(processor, EmbeddingProcessor)
            self.assertEqual(str(processor.cache_dir), temp_dir)

    def test_create_complete_system(self):
        """Test creating complete retrieval system."""
        system = RetrievalFactory.create_complete_retrieval_system()

        self.assertIn("retriever", system)
        self.assertIn("embedding_processor", system)
        self.assertIn("index_path", system)

        self.assertIsInstance(system["retriever"], CLIPRetriever)
        self.assertIsInstance(system["embedding_processor"], EmbeddingProcessor)

    def test_hardware_optimization(self):
        """Test hardware-optimized configuration."""
        config = RetrievalFactory.get_optimized_config_for_hardware()

        self.assertIsInstance(config, RetrievalConfig)
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.embedding_dim, 0)
        self.assertIn(config.device, ["cuda", "cpu"])

    def test_system_validation(self):
        """Test system requirements validation."""
        validation = RetrievalFactory.validate_system_requirements()

        self.assertIn("status", validation)
        self.assertIn("requirements_met", validation)
        self.assertIn("pytorch_version", validation)
        self.assertIn("cuda_available", validation)

        # Should not crash and return reasonable data
        self.assertIsInstance(validation["requirements_met"], bool)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete retrieval pipeline."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RetrievalConfig(device="cpu", batch_size=2)

    def tearDown(self):
        """Clean up after integration tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_images(self, count=5):
        """Create test images for integration testing."""
        image_dir = Path(self.temp_dir) / "images"
        image_dir.mkdir(exist_ok=True)

        image_paths = []
        for i in range(count):
            # Create different colored images
            color = (i * 50 % 255, (i * 70) % 255, (i * 90) % 255)
            image = Image.new('RGB', (224, 224), color=color)

            image_path = image_dir / f"test_image_{i}.jpg"
            image.save(image_path)
            image_paths.append(str(image_path))

        return image_paths

    @patch('src.retrieval.clip_retriever.CLIPModel')
    @patch('src.retrieval.clip_retriever.CLIPProcessor')
    def test_end_to_end_pipeline(self, mock_processor, mock_model):
        """Test complete end-to-end retrieval pipeline."""
        # Setup mocks
        mock_processor_instance = Mock()
        mock_model_instance = Mock()

        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock processor outputs
        mock_processor_instance.return_value = {
            "pixel_values": torch.randn(2, 3, 224, 224)
        }

        # Mock model outputs (different for images vs text)
        def mock_get_features(**kwargs):
            if "pixel_values" in kwargs:
                # Image features
                batch_size = kwargs["pixel_values"].shape[0]
                return torch.randn(batch_size, 512)
            else:
                # Text features
                batch_size = kwargs.get("input_ids", torch.tensor([[1]])).shape[0]
                return torch.randn(batch_size, 512)

        mock_model_instance.get_image_features.side_effect = mock_get_features
        mock_model_instance.get_text_features.side_effect = mock_get_features

        # Create test setup
        image_paths = self.create_test_images(count=5)

        # Create retrieval system
        retriever = RetrievalFactory.create_clip_retriever(self.config)
        processor = RetrievalFactory.create_embedding_processor(
            retriever, cache_dir=self.temp_dir
        )

        try:
            # Process images to generate embeddings
            embeddings, valid_paths = processor.process_image_corpus(
                image_paths, batch_size=2, max_images=5
            )

            # Verify embeddings were generated
            self.assertEqual(len(valid_paths), 5)
            self.assertEqual(embeddings.shape[0], 5)
            self.assertEqual(embeddings.shape[1], 512)

            # Build retrieval index
            retriever.build_index(embeddings, valid_paths)

            # Test retrieval
            results = retriever.retrieve_similar("test query", k=3)

            # Verify results
            self.assertLessEqual(len(results), 3)  # Should return at most k results

            # Test index save/load
            index_path = Path(self.temp_dir) / "test_index.bin"
            retriever.save_index(str(index_path))
            self.assertTrue(index_path.exists())

            # Create new retriever and load index
            new_retriever = RetrievalFactory.create_clip_retriever(self.config)
            new_retriever.load_index(str(index_path), valid_paths)

            # Test loaded index
            loaded_results = new_retriever.retrieve_similar("test query", k=3)
            self.assertLessEqual(len(loaded_results), 3)

        finally:
            # Clean up
            retriever.clear_memory()


def run_all_tests():
    """Run all unit tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestRetrievalConfig,
        TestRetrievalResult,
        TestCLIPRetriever,
        TestEmbeddingProcessor,
        TestRetrievalFactory,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)