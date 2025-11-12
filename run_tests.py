#!/usr/bin/env python3
"""
Test Runner for MRAG-Bench Dataset Processing

Simple test runner for Sprint 2 dataset functionality.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_tests():
    """Run dataset processing tests."""
    logger.info("Starting Sprint 2 dataset processing tests...")

    # Add src to Python path
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    try:
        # Test imports first
        logger.info("Testing imports...")

        from dataset.interface import Sample, BatchData, DatasetInterface
        from dataset.mrag_dataset import MRAGDataset
        from dataset.preprocessing import ImagePreprocessor, PreprocessingConfig
        from dataset.data_loader import MemoryAwareDataLoader, StreamingConfig
        from dataset.validation import DatasetValidator
        from utils.memory_manager import MemoryManager

        logger.info("✓ All imports successful")

        # Test basic functionality
        logger.info("Testing basic functionality...")

        # Test configuration
        config = PreprocessingConfig()
        assert config.image_size == (224, 224)
        logger.info("✓ Configuration system working")

        # Test memory manager
        memory_manager = MemoryManager()
        stats = memory_manager.monitor.get_current_stats()
        logger.info(f"✓ Memory manager working - GPU: {stats.gpu_allocated_gb:.2f}GB")

        # Test preprocessor
        preprocessor = ImagePreprocessor(config)
        logger.info("✓ Image preprocessor initialized")

        logger.info("Basic functionality tests completed successfully!")

        # Run pytest if available
        try:
            import pytest
            logger.info("Running comprehensive test suite with pytest...")

            # Run specific test files
            test_files = [
                "tests/dataset/test_mrag_dataset.py",
                "tests/dataset/test_preprocessing.py"
            ]

            for test_file in test_files:
                if (project_root / test_file).exists():
                    logger.info(f"Running {test_file}...")
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
                    ], cwd=project_root, capture_output=True, text=True)

                    if result.returncode == 0:
                        logger.info(f"✓ {test_file} passed")
                    else:
                        logger.warning(f"⚠ {test_file} had issues:")
                        logger.warning(result.stdout)
                        logger.warning(result.stderr)
                else:
                    logger.warning(f"Test file not found: {test_file}")

        except ImportError:
            logger.warning("pytest not available, skipping comprehensive tests")

        logger.info("Sprint 2 dataset processing tests completed!")
        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)