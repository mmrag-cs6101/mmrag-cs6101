#!/usr/bin/env python3
"""
Environment Setup Script for MRAG-Bench Reproduction System

This script sets up the development environment and validates system requirements.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Environment setup and validation for MRAG-Bench system."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_min_version = (3, 8)
        self.required_dirs = [
            "data", "data/mrag_bench", "data/embeddings",
            "results", "logs", "cache"
        ]

    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        current_version = sys.version_info[:2]
        if current_version < self.python_min_version:
            logger.error(
                f"Python {self.python_min_version[0]}.{self.python_min_version[1]}+ required, "
                f"but {current_version[0]}.{current_version[1]} is installed"
            )
            return False

        logger.info(f"Python version check passed: {current_version[0]}.{current_version[1]}")
        return True

    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and CUDA installation."""
        gpu_info = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_memory_gb": 0.0,
            "cuda_version": None,
            "recommendations": []
        }

        try:
            import torch
            gpu_info["cuda_available"] = torch.cuda.is_available()

            if gpu_info["cuda_available"]:
                gpu_info["gpu_count"] = torch.cuda.device_count()
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_info["gpu_memory_gb"] = gpu_props.total_memory / 1024**3
                gpu_info["cuda_version"] = torch.version.cuda

                logger.info(f"GPU detected: {gpu_props.name}")
                logger.info(f"GPU memory: {gpu_info['gpu_memory_gb']:.1f}GB")
                logger.info(f"CUDA version: {gpu_info['cuda_version']}")

                # Check memory requirements
                if gpu_info["gpu_memory_gb"] < 16:
                    gpu_info["recommendations"].append(
                        "GPU has less than 16GB VRAM - use aggressive quantization"
                    )
                elif gpu_info["gpu_memory_gb"] >= 16:
                    gpu_info["recommendations"].append("GPU memory sufficient for standard configuration")
            else:
                logger.warning("CUDA not available - will use CPU-only mode")
                gpu_info["recommendations"].append("Install CUDA for GPU acceleration")

        except ImportError:
            logger.warning("PyTorch not installed - cannot check GPU")
            gpu_info["recommendations"].append("Install PyTorch to check GPU availability")

        return gpu_info

    def create_directories(self) -> None:
        """Create required project directories."""
        logger.info("Creating project directories...")

        for dir_name in self.required_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

    def install_dependencies(self, install_dev: bool = True) -> bool:
        """Install Python dependencies from requirements.txt."""
        logger.info("Installing Python dependencies...")

        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            return False

        try:
            # Install main dependencies
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Dependencies installed successfully")

            # Install development dependencies if requested
            if install_dev:
                dev_packages = [
                    "pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0",
                    "mypy>=1.0.0", "pytest-cov>=4.0.0"
                ]
                cmd_dev = [sys.executable, "-m", "pip", "install"] + dev_packages
                subprocess.run(cmd_dev, capture_output=True, text=True, check=True)
                logger.info("Development dependencies installed")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def validate_installation(self) -> Dict[str, bool]:
        """Validate that key packages are properly installed."""
        logger.info("Validating package installation...")

        validation_results = {}

        # Core packages to validate
        core_packages = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("PIL", "Pillow"),
            ("numpy", "NumPy"),
            ("yaml", "PyYAML"),
            ("faiss", "FAISS")
        ]

        for package, name in core_packages:
            try:
                __import__(package)
                validation_results[name] = True
                logger.debug(f"✓ {name} imported successfully")
            except ImportError:
                validation_results[name] = False
                logger.warning(f"✗ {name} import failed")

        return validation_results

    def create_config_file(self) -> None:
        """Create default configuration file if it doesn't exist."""
        config_file = self.project_root / "config" / "mrag_bench.yaml"

        if config_file.exists():
            logger.info("Configuration file already exists")
            return

        logger.info("Creating default configuration file...")

        # Import after potential installation
        try:
            from src.config import create_default_config_file
            create_default_config_file(str(config_file))
            logger.info(f"Created configuration file: {config_file}")
        except ImportError:
            logger.warning("Could not create config file - source not available")

    def run_basic_tests(self) -> bool:
        """Run basic system tests to validate setup."""
        logger.info("Running basic system tests...")

        try:
            # Test configuration loading
            from src.config import MRAGConfig
            config = MRAGConfig()
            config.validate()
            logger.debug("✓ Configuration system test passed")

            # Test memory monitoring
            from src.utils import MemoryMonitor
            monitor = MemoryMonitor()
            stats = monitor.get_current_stats()
            logger.debug(f"✓ Memory monitoring test passed - GPU: {stats.gpu_total_gb:.1f}GB")

            return True

        except Exception as e:
            logger.error(f"Basic tests failed: {e}")
            return False

    def setup(self, install_deps: bool = True, run_tests: bool = True) -> bool:
        """Run complete environment setup."""
        logger.info("Starting MRAG-Bench environment setup...")

        # Step 1: Check Python version
        if not self.check_python_version():
            return False

        # Step 2: Create directories
        self.create_directories()

        # Step 3: Check GPU availability
        gpu_info = self.check_gpu_availability()

        # Step 4: Install dependencies
        if install_deps:
            if not self.install_dependencies():
                return False

            # Validate installation
            validation_results = self.validate_installation()
            failed_packages = [pkg for pkg, success in validation_results.items() if not success]

            if failed_packages:
                logger.warning(f"Some packages failed validation: {failed_packages}")
                logger.warning("System may not function correctly")

        # Step 5: Create configuration
        self.create_config_file()

        # Step 6: Run basic tests
        if run_tests:
            if not self.run_basic_tests():
                logger.warning("Some basic tests failed - check configuration")

        # Step 7: Display summary
        self._display_setup_summary(gpu_info)

        logger.info("Environment setup completed!")
        return True

    def _display_setup_summary(self, gpu_info: Dict[str, Any]) -> None:
        """Display setup summary and recommendations."""
        print("\n" + "="*60)
        print("MRAG-BENCH ENVIRONMENT SETUP SUMMARY")
        print("="*60)

        print(f"Project Root: {self.project_root}")
        print(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

        if gpu_info["cuda_available"]:
            print(f"GPU: Available ({gpu_info['gpu_memory_gb']:.1f}GB VRAM)")
            print(f"CUDA Version: {gpu_info['cuda_version']}")
        else:
            print("GPU: Not available (CPU-only mode)")

        print("\nNext Steps:")
        print("1. Download MRAG-Bench dataset to data/mrag_bench/")
        print("2. Review configuration in config/mrag_bench.yaml")
        print("3. Run: python -m pytest tests/ (if tests are implemented)")
        print("4. Start with Sprint 2 dataset processing")

        if gpu_info["recommendations"]:
            print("\nRecommendations:")
            for rec in gpu_info["recommendations"]:
                print(f"- {rec}")

        print("\n" + "="*60)


def main():
    """Main entry point for environment setup."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup MRAG-Bench environment")
    parser.add_argument("--no-install", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--no-tests", action="store_true",
                       help="Skip basic tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup = EnvironmentSetup()
    success = setup.setup(
        install_deps=not args.no_install,
        run_tests=not args.no_tests
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()