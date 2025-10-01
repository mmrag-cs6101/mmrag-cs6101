#!/usr/bin/env python3
"""
Test CLIP Retrieval System

Comprehensive test script for Sprint 3 CLIP-based image retrieval implementation.
Tests embedding generation, FAISS indexing, and similarity search functionality.
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_default_config
from src.dataset.mrag_dataset import MRAGDataset
from src.retrieval import RetrievalFactory, CLIPRetriever, EmbeddingProcessor
from src.utils.memory_manager import MemoryManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_clip_retrieval.log')
    ]
)

logger = logging.getLogger(__name__)


class CLIPRetrievalTester:
    """Comprehensive tester for CLIP retrieval system."""

    def __init__(self, config_path: str = None, test_mode: str = "full"):
        """
        Initialize tester.

        Args:
            config_path: Path to configuration file
            test_mode: Test mode ('quick', 'full', 'stress')
        """
        self.test_mode = test_mode
        self.config = get_default_config()

        if config_path and os.path.exists(config_path):
            self.config = self.config.load(config_path)

        self.results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {},
            "errors": []
        }

        logger.info(f"CLIPRetrievalTester initialized in {test_mode} mode")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive CLIP retrieval tests...")

        try:
            # Test 1: System validation
            self._test_system_requirements()

            # Test 2: Hardware optimization
            self._test_hardware_optimization()

            # Test 3: CLIP model loading
            self._test_clip_model_loading()

            # Test 4: Embedding generation
            self._test_embedding_generation()

            # Test 5: FAISS indexing
            self._test_faiss_indexing()

            # Test 6: Similarity search
            self._test_similarity_search()

            # Test 7: End-to-end retrieval
            self._test_end_to_end_retrieval()

            # Test 8: Performance benchmarking
            if self.test_mode in ["full", "stress"]:
                self._test_performance_benchmarks()

            # Test 9: Memory management
            self._test_memory_management()

            # Test 10: Stress test (if enabled)
            if self.test_mode == "stress":
                self._test_stress_scenarios()

        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.results["errors"].append(f"Test suite failure: {str(e)}")

        # Generate final report
        self._generate_test_report()

        return self.results

    def _test_system_requirements(self):
        """Test system requirements validation."""
        logger.info("Testing system requirements...")

        try:
            validation = RetrievalFactory.validate_system_requirements()

            if validation["requirements_met"]:
                self._pass_test("System requirements validation")
                logger.info(f"System validation: {validation['status']}")
            else:
                self._fail_test("System requirements validation", validation["errors"])

            self.results["performance_metrics"]["system_validation"] = validation

        except Exception as e:
            self._fail_test("System requirements validation", str(e))

    def _test_hardware_optimization(self):
        """Test hardware-optimized configuration."""
        logger.info("Testing hardware optimization...")

        try:
            optimized_config = RetrievalFactory.get_optimized_config_for_hardware()

            # Validate configuration
            if optimized_config.embedding_dim > 0 and optimized_config.batch_size > 0:
                self._pass_test("Hardware optimization")
                logger.info(f"Optimized config: batch_size={optimized_config.batch_size}, device={optimized_config.device}")
            else:
                self._fail_test("Hardware optimization", "Invalid optimized configuration")

            self.results["performance_metrics"]["optimized_config"] = {
                "batch_size": optimized_config.batch_size,
                "device": optimized_config.device,
                "max_memory_gb": optimized_config.max_memory_gb
            }

        except Exception as e:
            self._fail_test("Hardware optimization", str(e))

    def _test_clip_model_loading(self):
        """Test CLIP model loading and initialization."""
        logger.info("Testing CLIP model loading...")

        try:
            retriever = RetrievalFactory.create_clip_retriever(mrag_config=self.config)

            # Test model warmup
            warmup_stats = retriever.warmup()

            if warmup_stats and "image_encoding_time" in warmup_stats:
                self._pass_test("CLIP model loading")
                logger.info(f"Model warmup completed: {warmup_stats}")
            else:
                self._fail_test("CLIP model loading", "Warmup failed")

            self.results["performance_metrics"]["model_warmup"] = warmup_stats

            # Test memory usage
            memory_stats = retriever.get_memory_usage()
            logger.info(f"CLIP memory usage: {memory_stats}")

            # Clean up
            retriever.clear_memory()

        except Exception as e:
            self._fail_test("CLIP model loading", str(e))

    def _test_embedding_generation(self):
        """Test image and text embedding generation."""
        logger.info("Testing embedding generation...")

        try:
            retriever = RetrievalFactory.create_clip_retriever(mrag_config=self.config)

            # Test image embedding
            from PIL import Image
            test_images = [
                Image.new('RGB', (224, 224), color=(255, 0, 0)),
                Image.new('RGB', (224, 224), color=(0, 255, 0)),
                Image.new('RGB', (224, 224), color=(0, 0, 255))
            ]

            start_time = time.time()
            image_embeddings = retriever.encode_images(test_images)
            image_time = time.time() - start_time

            # Test text embedding
            test_texts = ["red medical image", "green medical scan", "blue diagnostic image"]

            start_time = time.time()
            text_embeddings = retriever.encode_text(test_texts)
            text_time = time.time() - start_time

            # Validate embeddings
            if (image_embeddings.shape == (3, 512) and
                text_embeddings.shape == (3, 512)):
                self._pass_test("Embedding generation")
                logger.info(f"Embedding generation successful - Images: {image_time:.3f}s, Text: {text_time:.3f}s")
            else:
                self._fail_test("Embedding generation", f"Wrong embedding shapes: {image_embeddings.shape}, {text_embeddings.shape}")

            self.results["performance_metrics"]["embedding_generation"] = {
                "image_encoding_time": image_time,
                "text_encoding_time": text_time,
                "image_shape": image_embeddings.shape,
                "text_shape": text_embeddings.shape
            }

            # Clean up
            retriever.clear_memory()

        except Exception as e:
            self._fail_test("Embedding generation", str(e))

    def _test_faiss_indexing(self):
        """Test FAISS index building and operations."""
        logger.info("Testing FAISS indexing...")

        try:
            retriever = RetrievalFactory.create_clip_retriever(mrag_config=self.config)

            # Generate sample embeddings
            import numpy as np
            num_samples = 100 if self.test_mode == "quick" else 1000
            sample_embeddings = np.random.randn(num_samples, 512).astype(np.float32)

            # Normalize embeddings
            sample_embeddings = sample_embeddings / np.linalg.norm(sample_embeddings, axis=1, keepdims=True)

            sample_paths = [f"test_image_{i}.jpg" for i in range(num_samples)]

            # Build index
            start_time = time.time()
            retriever.build_index(sample_embeddings, sample_paths)
            index_time = time.time() - start_time

            # Test index operations
            if retriever.index is not None and retriever.index.ntotal == num_samples:
                self._pass_test("FAISS indexing")
                logger.info(f"FAISS index built successfully in {index_time:.3f}s for {num_samples} embeddings")
            else:
                self._fail_test("FAISS indexing", f"Index building failed: {retriever.index}")

            self.results["performance_metrics"]["faiss_indexing"] = {
                "index_build_time": index_time,
                "num_embeddings": num_samples,
                "index_type": type(retriever.index).__name__ if retriever.index else None
            }

            # Test index saving/loading
            test_index_path = "test_faiss_index.bin"
            try:
                retriever.save_index(test_index_path)

                # Create new retriever and load index
                new_retriever = RetrievalFactory.create_clip_retriever(mrag_config=self.config)
                new_retriever.load_index(test_index_path, sample_paths)

                if new_retriever.index.ntotal == num_samples:
                    logger.info("Index save/load test successful")
                else:
                    logger.warning("Index save/load test failed")

                # Clean up test file
                if os.path.exists(test_index_path):
                    os.remove(test_index_path)

            except Exception as e:
                logger.warning(f"Index save/load test failed: {e}")

            # Clean up
            retriever.clear_memory()

        except Exception as e:
            self._fail_test("FAISS indexing", str(e))

    def _test_similarity_search(self):
        """Test similarity search functionality."""
        logger.info("Testing similarity search...")

        try:
            retriever = RetrievalFactory.create_clip_retriever(mrag_config=self.config)

            # Build test index
            import numpy as np
            num_samples = 50
            sample_embeddings = np.random.randn(num_samples, 512).astype(np.float32)
            sample_embeddings = sample_embeddings / np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
            sample_paths = [f"test_image_{i}.jpg" for i in range(num_samples)]

            retriever.build_index(sample_embeddings, sample_paths)

            # Test similarity search
            test_queries = ["medical image", "diagnostic scan", "red blood cell"]

            search_times = []
            for query in test_queries:
                start_time = time.time()
                results = retriever.retrieve_similar(query, k=5)
                search_time = time.time() - start_time
                search_times.append(search_time)

                if len(results) > 0:
                    logger.info(f"Query '{query}': {len(results)} results, top score: {results[0].similarity_score:.3f}")
                else:
                    logger.warning(f"No results for query '{query}'")

            avg_search_time = sum(search_times) / len(search_times)

            if avg_search_time < 5.0:  # Target: <5 seconds
                self._pass_test("Similarity search")
                logger.info(f"Similarity search successful - Avg time: {avg_search_time:.3f}s")
            else:
                self._fail_test("Similarity search", f"Search too slow: {avg_search_time:.3f}s > 5.0s target")

            self.results["performance_metrics"]["similarity_search"] = {
                "avg_search_time": avg_search_time,
                "search_times": search_times,
                "target_met": avg_search_time < 5.0
            }

            # Clean up
            retriever.clear_memory()

        except Exception as e:
            self._fail_test("Similarity search", str(e))

    def _test_end_to_end_retrieval(self):
        """Test complete end-to-end retrieval pipeline."""
        logger.info("Testing end-to-end retrieval...")

        try:
            # Create complete retrieval system
            system = RetrievalFactory.create_complete_retrieval_system(mrag_config=self.config)
            retriever = system["retriever"]
            embedding_processor = system["embedding_processor"]

            # Test with dummy dataset
            from PIL import Image
            test_images = [
                Image.new('RGB', (224, 224), color=(255, 0, 0)),
                Image.new('RGB', (224, 224), color=(0, 255, 0)),
                Image.new('RGB', (224, 224), color=(0, 0, 255)),
                Image.new('RGB', (224, 224), color=(255, 255, 0))
            ]

            # Save test images temporarily
            test_dir = Path("test_images")
            test_dir.mkdir(exist_ok=True)

            test_paths = []
            for i, img in enumerate(test_images):
                path = test_dir / f"test_{i}.jpg"
                img.save(path)
                test_paths.append(str(path))

            try:
                # Process embeddings
                start_time = time.time()
                embeddings, valid_paths = embedding_processor.process_image_corpus(
                    test_paths, batch_size=2, max_images=4
                )
                processing_time = time.time() - start_time

                # Build index
                retriever.build_index(embeddings, valid_paths)

                # Test retrieval
                results = retriever.retrieve_similar("red medical image", k=2)

                if len(results) > 0 and processing_time < 30:  # Target: <30s total
                    self._pass_test("End-to-end retrieval")
                    logger.info(f"End-to-end test successful - Processing: {processing_time:.3f}s")
                else:
                    self._fail_test("End-to-end retrieval", f"Test failed: {len(results)} results, {processing_time:.3f}s")

                self.results["performance_metrics"]["end_to_end"] = {
                    "processing_time": processing_time,
                    "num_results": len(results),
                    "target_met": processing_time < 30
                }

            finally:
                # Clean up test images
                for path in test_paths:
                    if os.path.exists(path):
                        os.remove(path)
                if test_dir.exists():
                    test_dir.rmdir()

            # Clean up
            retriever.clear_memory()

        except Exception as e:
            self._fail_test("End-to-end retrieval", str(e))

    def _test_performance_benchmarks(self):
        """Test performance benchmarks and optimization."""
        logger.info("Testing performance benchmarks...")

        try:
            retriever = RetrievalFactory.create_clip_retriever(mrag_config=self.config)

            # Benchmark embedding generation
            from PIL import Image
            batch_sizes = [1, 4, 8, 16, 32]
            embedding_times = {}

            for batch_size in batch_sizes:
                if batch_size > 16 and self.test_mode == "quick":
                    continue

                test_images = [
                    Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                    for i in range(batch_size)
                ]

                start_time = time.time()
                embeddings = retriever.encode_images(test_images)
                batch_time = time.time() - start_time

                embedding_times[batch_size] = {
                    "total_time": batch_time,
                    "per_image_time": batch_time / batch_size
                }

                logger.info(f"Batch size {batch_size}: {batch_time:.3f}s total, {batch_time/batch_size:.3f}s per image")

            # Find optimal batch size (best throughput)
            optimal_batch = min(embedding_times.keys(),
                              key=lambda x: embedding_times[x]["per_image_time"])

            self._pass_test("Performance benchmarks")
            logger.info(f"Optimal batch size: {optimal_batch}")

            self.results["performance_metrics"]["benchmarks"] = {
                "embedding_times": embedding_times,
                "optimal_batch_size": optimal_batch
            }

            # Clean up
            retriever.clear_memory()

        except Exception as e:
            self._fail_test("Performance benchmarks", str(e))

    def _test_memory_management(self):
        """Test memory management and optimization."""
        logger.info("Testing memory management...")

        try:
            memory_manager = MemoryManager(memory_limit_gb=15.0)
            initial_stats = memory_manager.monitor.get_current_stats()

            # Test memory monitoring
            retriever = RetrievalFactory.create_clip_retriever(mrag_config=self.config)

            with memory_manager.memory_guard("Memory test"):
                # Load model and check memory
                retriever._load_model()
                model_stats = memory_manager.monitor.get_current_stats()

                # Test memory clearing
                retriever.clear_memory()
                cleared_stats = memory_manager.monitor.get_current_stats()

                memory_increase = model_stats.gpu_allocated_gb - initial_stats.gpu_allocated_gb
                memory_cleared = model_stats.gpu_allocated_gb - cleared_stats.gpu_allocated_gb

                if memory_increase > 0 and memory_cleared > 0:
                    self._pass_test("Memory management")
                    logger.info(f"Memory test: +{memory_increase:.2f}GB loading, -{memory_cleared:.2f}GB clearing")
                else:
                    self._fail_test("Memory management", "Memory tracking failed")

                self.results["performance_metrics"]["memory_management"] = {
                    "initial_memory_gb": initial_stats.gpu_allocated_gb,
                    "peak_memory_gb": model_stats.gpu_allocated_gb,
                    "final_memory_gb": cleared_stats.gpu_allocated_gb,
                    "memory_increase_gb": memory_increase,
                    "memory_cleared_gb": memory_cleared
                }

        except Exception as e:
            self._fail_test("Memory management", str(e))

    def _test_stress_scenarios(self):
        """Test stress scenarios and edge cases."""
        logger.info("Testing stress scenarios...")

        try:
            retriever = RetrievalFactory.create_clip_retriever(mrag_config=self.config)

            # Stress test: Large batch processing
            from PIL import Image
            large_batch = [
                Image.new('RGB', (224, 224), color=(i*10, i*10, i*10))
                for i in range(50)  # Large batch
            ]

            start_time = time.time()
            embeddings = retriever.encode_images(large_batch)
            large_batch_time = time.time() - start_time

            # Stress test: Rapid sequential queries
            import numpy as np
            num_test_embeddings = 1000
            test_embeddings = np.random.randn(num_test_embeddings, 512).astype(np.float32)
            test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
            test_paths = [f"stress_test_{i}.jpg" for i in range(num_test_embeddings)]

            retriever.build_index(test_embeddings, test_paths)

            rapid_queries = ["query " + str(i) for i in range(20)]
            rapid_start = time.time()

            for query in rapid_queries:
                results = retriever.retrieve_similar(query, k=5)

            rapid_time = time.time() - rapid_start

            if large_batch_time < 60 and rapid_time < 10:  # Reasonable thresholds
                self._pass_test("Stress scenarios")
                logger.info(f"Stress test passed - Large batch: {large_batch_time:.1f}s, Rapid queries: {rapid_time:.1f}s")
            else:
                self._fail_test("Stress scenarios", f"Performance too slow: {large_batch_time:.1f}s, {rapid_time:.1f}s")

            self.results["performance_metrics"]["stress_test"] = {
                "large_batch_time": large_batch_time,
                "rapid_queries_time": rapid_time,
                "large_batch_size": len(large_batch),
                "num_rapid_queries": len(rapid_queries)
            }

            # Clean up
            retriever.clear_memory()

        except Exception as e:
            self._fail_test("Stress scenarios", str(e))

    def _pass_test(self, test_name: str):
        """Record a passed test."""
        self.results["tests_passed"] += 1
        logger.info(f"✓ {test_name} - PASSED")

    def _fail_test(self, test_name: str, error: str):
        """Record a failed test."""
        self.results["tests_failed"] += 1
        self.results["errors"].append(f"{test_name}: {error}")
        logger.error(f"✗ {test_name} - FAILED: {error}")

    def _generate_test_report(self):
        """Generate comprehensive test report."""
        total_tests = self.results["tests_passed"] + self.results["tests_failed"]
        pass_rate = self.results["tests_passed"] / total_tests * 100 if total_tests > 0 else 0

        report = f"""
=== CLIP Retrieval System Test Report ===

Test Mode: {self.test_mode}
Total Tests: {total_tests}
Passed: {self.results["tests_passed"]}
Failed: {self.results["tests_failed"]}
Pass Rate: {pass_rate:.1f}%

Performance Metrics:
"""

        for metric, data in self.results["performance_metrics"].items():
            report += f"  {metric}: {data}\n"

        if self.results["errors"]:
            report += "\nErrors:\n"
            for error in self.results["errors"]:
                report += f"  - {error}\n"

        # Overall assessment
        if pass_rate >= 90:
            report += "\n🎉 OVERALL: EXCELLENT - System ready for production\n"
        elif pass_rate >= 70:
            report += "\n✅ OVERALL: GOOD - Minor issues to address\n"
        elif pass_rate >= 50:
            report += "\n⚠️  OVERALL: ACCEPTABLE - Several issues need fixing\n"
        else:
            report += "\n❌ OVERALL: POOR - Major issues require attention\n"

        logger.info(report)

        # Save report to file
        with open("clip_retrieval_test_report.txt", "w") as f:
            f.write(report)

        return report


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test CLIP Retrieval System")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "stress"],
        default="full",
        help="Test mode: quick (fast tests), full (comprehensive), stress (performance limits)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting CLIP retrieval system tests in {args.mode} mode...")

    # Run tests
    tester = CLIPRetrievalTester(config_path=args.config, test_mode=args.mode)
    results = tester.run_all_tests()

    # Exit with appropriate code
    exit_code = 0 if results["tests_failed"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()