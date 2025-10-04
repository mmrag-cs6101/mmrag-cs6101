"""
End-to-End MRAG Pipeline

Orchestrates retrieval and generation for complete multimodal RAG system.
Integrates CLIP retrieval with LLaVA generation with memory management.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from .config import MRAGConfig
from .dataset import MRAGDataset, Sample
from .retrieval import CLIPRetriever, RetrievalConfig
from .generation import LLaVAGenerationPipeline, GenerationConfig, MultimodalContext, GenerationResult
from .utils.memory_manager import MemoryManager
from .utils.error_handling import handle_errors, MRAGError, ErrorCategory, ErrorSeverity


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline result with retrieval and generation components."""
    question_id: str
    question: str
    retrieved_images: List[str]  # Image paths
    retrieval_scores: List[float]
    generated_answer: str
    confidence_score: float
    total_time: float
    retrieval_time: float
    generation_time: float
    memory_usage: Dict[str, float]
    metadata: Dict[str, Any]

    # Sprint 6 Enhancement: Additional monitoring fields
    pipeline_stage_times: Dict[str, float] = None
    memory_usage_per_stage: Dict[str, Dict[str, float]] = None
    error_recovery_attempts: int = 0
    optimization_triggers: List[str] = None

    def __post_init__(self):
        """Initialize default values for new fields."""
        if self.pipeline_stage_times is None:
            self.pipeline_stage_times = {}
        if self.memory_usage_per_stage is None:
            self.memory_usage_per_stage = {}
        if self.optimization_triggers is None:
            self.optimization_triggers = []


class MRAGPipeline:
    """
    Complete MRAG-Bench pipeline orchestrating retrieval and generation.

    Features:
    - Sequential model loading to minimize memory overlap
    - Dynamic memory management between pipeline stages
    - Comprehensive error handling and recovery
    - Performance monitoring and optimization
    - Memory pressure detection and automatic cleanup
    """

    def __init__(self, config: MRAGConfig):
        """
        Initialize MRAG pipeline.

        Args:
            config: Complete MRAG system configuration
        """
        self.config = config
        self.memory_manager = MemoryManager(
            memory_limit_gb=config.performance.memory_limit_gb,
            buffer_gb=config.performance.memory_buffer_gb
        )

        # Pipeline components (initialized on-demand)
        self.dataset = None
        self.retriever = None
        self.generator = None

        # Component loading states
        self.retriever_loaded = False
        self.generator_loaded = False

        # Performance tracking
        self.pipeline_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_total_time": 0.0,
            "memory_optimizations_triggered": 0,
            "error_recoveries_performed": 0
        }

        # Sprint 6 Enhancement: Performance monitoring and optimization
        self.performance_monitor = {
            "retrieval_time_threshold": config.performance.retrieval_timeout,
            "generation_time_threshold": config.performance.generation_timeout,
            "total_time_threshold": config.performance.total_pipeline_timeout,
            "memory_optimization_threshold": 0.9  # 90% of memory limit
        }

        # Sprint 6 Enhancement: Error recovery strategies
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3

        logger.info("MRAG pipeline initialized with Sprint 6 enhancements")

    @handle_errors
    def initialize_dataset(self) -> None:
        """Initialize dataset component."""
        if self.dataset is not None:
            return

        logger.info("Initializing MRAG dataset...")

        self.dataset = MRAGDataset(
            data_path=self.config.dataset.data_path,
            batch_size=self.config.dataset.batch_size,
            image_size=self.config.dataset.image_size
        )

        # Validate dataset
        validation_results = self.dataset.validate_dataset()
        if validation_results["status"] == "error":
            raise MRAGError(f"Dataset validation failed: {validation_results['errors']}")

        logger.info(f"Dataset initialized with {validation_results['total_samples']} samples")

    def _encode_corpus_in_batches(self, corpus_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode corpus images in batches to avoid memory issues.

        Args:
            corpus_paths: List of image paths to encode
            batch_size: Number of images to process per batch

        Returns:
            Numpy array of embeddings
        """
        import numpy as np
        from PIL import Image

        all_embeddings = []
        total_batches = (len(corpus_paths) + batch_size - 1) // batch_size

        logger.info(f"Encoding {len(corpus_paths)} images in {total_batches} batches of {batch_size}...")

        for batch_idx in range(0, len(corpus_paths), batch_size):
            batch_paths = corpus_paths[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"Processing batch {batch_num}/{total_batches}...")

            # Load batch images
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    # Use a blank image as placeholder
                    batch_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

            # Encode batch
            batch_embeddings = self.retriever.encode_images(batch_images)
            all_embeddings.append(batch_embeddings)

            # Clean up
            del batch_images
            if batch_num % 50 == 0:
                import gc
                gc.collect()

        logger.info("Concatenating embeddings...")
        return np.vstack(all_embeddings)

    @handle_errors
    def load_retriever(self) -> None:
        """Load CLIP retrieval component."""
        if self.retriever_loaded:
            return

        logger.info("Loading CLIP retriever...")

        with self.memory_manager.memory_guard("CLIP retriever loading"):
            # Create retrieval configuration
            retrieval_config = RetrievalConfig(
                model_name=self.config.model.retriever_name,
                embedding_dim=self.config.retrieval.embedding_dim,
                top_k=self.config.retrieval.top_k,
                similarity_threshold=self.config.retrieval.similarity_threshold,
                batch_size=self.config.retrieval.batch_size,
                device=self.config.model.device
            )

            self.retriever = CLIPRetriever(retrieval_config)

            # Build index if not cached
            # Note: Model loads automatically when needed (lazy loading)
            if self.dataset is None:
                self.initialize_dataset()

            # Get corpus image paths and encode them
            corpus_paths = self.dataset.get_retrieval_corpus()
            logger.info(f"Encoding {len(corpus_paths)} corpus images...")

            # Check for cached embeddings
            import os
            import numpy as np
            cache_dir = Path(self.config.dataset.embedding_cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            embeddings_cache_path = cache_dir / "corpus_embeddings.npy"
            paths_cache_path = cache_dir / "corpus_paths.txt"

            if embeddings_cache_path.exists() and paths_cache_path.exists():
                logger.info("Loading cached embeddings...")
                embeddings = np.load(embeddings_cache_path)
                with open(paths_cache_path, 'r') as f:
                    cached_paths = [line.strip() for line in f]

                # Verify cache matches current corpus
                if cached_paths == corpus_paths:
                    logger.info(f"Using cached embeddings for {len(corpus_paths)} images")
                else:
                    logger.warning("Cache mismatch, re-encoding corpus...")
                    embeddings = self._encode_corpus_in_batches(corpus_paths)
                    np.save(embeddings_cache_path, embeddings)
                    with open(paths_cache_path, 'w') as f:
                        f.write('\n'.join(corpus_paths))
            else:
                logger.info("No cache found, encoding corpus in batches...")
                embeddings = self._encode_corpus_in_batches(corpus_paths)

                # Save cache
                logger.info("Saving embeddings cache...")
                np.save(embeddings_cache_path, embeddings)
                with open(paths_cache_path, 'w') as f:
                    f.write('\n'.join(corpus_paths))

            # Build FAISS index
            self.retriever.build_index(embeddings, corpus_paths)

            self.retriever_loaded = True
            memory_stats = self.memory_manager.monitor.log_memory_stats("After CLIP loading")

        logger.info(f"CLIP retriever loaded successfully")

    @handle_errors
    def load_generator(self) -> None:
        """Load LLaVA generation component."""
        if self.generator_loaded:
            return

        logger.info("Loading LLaVA generator...")

        with self.memory_manager.memory_guard("LLaVA generator loading"):
            # Create generation configuration
            generation_config = GenerationConfig(
                model_name=self.config.model.vlm_name,
                max_length=self.config.generation.max_length,
                temperature=self.config.generation.temperature,
                do_sample=self.config.generation.do_sample,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                quantization=self.config.model.quantization,
                max_memory_gb=self.config.model.max_memory_gb,
                device=self.config.model.device,
                torch_dtype=self.config.model.torch_dtype
            )

            self.generator = LLaVAGenerationPipeline(generation_config)
            self.generator.load_model()

            self.generator_loaded = True
            memory_stats = self.memory_manager.monitor.log_memory_stats("After LLaVA loading")

        logger.info(f"LLaVA generator loaded successfully")

    @handle_errors
    def unload_retriever(self) -> None:
        """Unload retriever to free memory."""
        if not self.retriever_loaded:
            return

        logger.info("Unloading CLIP retriever...")
        if self.retriever is not None:
            self.retriever.unload_model()
            self.retriever = None

        self.retriever_loaded = False
        self.memory_manager.clear_gpu_memory(aggressive=True)

        logger.info("CLIP retriever unloaded")

    @handle_errors
    def unload_generator(self) -> None:
        """Unload generator to free memory."""
        if not self.generator_loaded:
            return

        logger.info("Unloading LLaVA generator...")
        if self.generator is not None:
            self.generator.unload_model()
            self.generator = None

        self.generator_loaded = False
        self.memory_manager.clear_gpu_memory(aggressive=True)

        logger.info("LLaVA generator unloaded")

    @handle_errors
    def process_query(
        self,
        question: str,
        question_id: str = "",
        ground_truth: str = "",
        use_sequential_loading: bool = True
    ) -> PipelineResult:
        """
        Process a single query through the complete pipeline.

        Args:
            question: Question text
            question_id: Unique question identifier
            ground_truth: Ground truth answer (for evaluation)
            use_sequential_loading: If True, load/unload models sequentially to save memory

        Returns:
            Complete pipeline result
        """
        start_time = time.time()
        retrieval_time = 0.0
        generation_time = 0.0

        try:
            # Initialize dataset if needed
            if self.dataset is None:
                self.initialize_dataset()

            # Step 1: Image Retrieval
            logger.debug(f"Processing query: {question[:100]}...")

            if use_sequential_loading:
                # Load only retriever
                self.load_retriever()
                if self.generator_loaded:
                    self.unload_generator()

            retrieval_start = time.time()

            if not self.retriever_loaded:
                self.load_retriever()

            # Perform retrieval
            retrieval_results = self.retriever.retrieve_similar(
                question,
                k=self.config.retrieval.top_k
            )

            retrieval_time = time.time() - retrieval_start

            # Load retrieved images
            retrieved_images = []
            retrieval_scores = []
            image_paths = []

            for result in retrieval_results:
                try:
                    image = Image.open(result.image_path).convert('RGB')
                    retrieved_images.append(image)
                    retrieval_scores.append(result.similarity_score)
                    image_paths.append(result.image_path)
                except Exception as e:
                    logger.warning(f"Failed to load retrieved image {result.image_path}: {e}")
                    continue

            logger.info(f"Retrieved {len(retrieved_images)} images in {retrieval_time:.2f}s")

            # Step 2: Answer Generation
            if use_sequential_loading:
                # Switch to generator
                self.unload_retriever()
                self.load_generator()

            generation_start = time.time()

            if not self.generator_loaded:
                self.load_generator()

            # Create multimodal context
            context = MultimodalContext(
                question=question,
                images=retrieved_images,
                image_paths=image_paths,
                context_metadata={
                    "question_id": question_id,
                    "ground_truth": ground_truth,
                    "retrieval_scores": retrieval_scores
                }
            )

            # Generate answer
            generation_result = self.generator.generate_answer(context)
            generation_time = time.time() - generation_start

            total_time = time.time() - start_time

            logger.info(
                f"Generated answer in {generation_time:.2f}s "
                f"(total: {total_time:.2f}s)"
            )

            # Sprint 6 Enhancement: Performance monitoring and optimization
            optimization_triggers = self.check_performance_triggers(
                retrieval_time, generation_time, total_time
            )

            if optimization_triggers:
                self.apply_performance_optimizations(optimization_triggers)

            # Update statistics
            self._update_stats(retrieval_time, generation_time, total_time)

            # Sprint 6 Enhancement: Detailed performance tracking
            stage_times = {
                "dataset_init": 0.0,  # Would need to track this separately
                "model_loading": 0.0,  # Would need to track this separately
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": total_time
            }

            memory_usage_per_stage = {
                "after_retrieval": self.memory_manager.monitor.get_current_stats().__dict__,
                "after_generation": self.memory_manager.monitor.get_current_stats().__dict__
            }

            # Create enhanced pipeline result
            result = PipelineResult(
                question_id=question_id,
                question=question,
                retrieved_images=image_paths,
                retrieval_scores=retrieval_scores,
                generated_answer=generation_result.answer,
                confidence_score=generation_result.confidence_score,
                total_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                memory_usage=self.memory_manager.monitor.get_current_stats().__dict__,
                metadata={
                    "ground_truth": ground_truth,
                    "num_retrieved_images": len(retrieved_images),
                    "generation_metadata": generation_result.metadata
                },
                # Sprint 6 enhancements
                pipeline_stage_times=stage_times,
                memory_usage_per_stage=memory_usage_per_stage,
                error_recovery_attempts=0,
                optimization_triggers=optimization_triggers
            )

            return result

        except Exception as e:
            logger.error(f"Pipeline processing failed for question {question_id}: {e}")

            # Sprint 6 Enhancement: Error recovery attempt
            error_stage = self._identify_error_stage(e, retrieval_time, generation_time)
            recovery_successful = self.handle_pipeline_error(e, error_stage, question_id)

            error_recovery_attempts = self.recovery_attempts.get(f"{error_stage}:{question_id}", 0)

            if recovery_successful and error_recovery_attempts < self.max_recovery_attempts:
                logger.info(f"Error recovery successful, retrying query {question_id}")
                # Recursive retry with recovery applied
                return self.process_query(question, question_id, ground_truth, use_sequential_loading)

            self.memory_manager.emergency_cleanup()
            self.pipeline_stats["failed_queries"] += 1

            # Return enhanced error result
            return PipelineResult(
                question_id=question_id,
                question=question,
                retrieved_images=[],
                retrieval_scores=[],
                generated_answer="Error: Unable to process query",
                confidence_score=0.0,
                total_time=time.time() - start_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                memory_usage=self.memory_manager.monitor.get_current_stats().__dict__,
                metadata={"error": str(e), "error_stage": error_stage},
                # Sprint 6 enhancements
                pipeline_stage_times={"error_occurred_at": time.time() - start_time},
                memory_usage_per_stage={"error": self.memory_manager.monitor.get_current_stats().__dict__},
                error_recovery_attempts=error_recovery_attempts,
                optimization_triggers=[]
            )

    def _identify_error_stage(self, error: Exception, retrieval_time: float, generation_time: float) -> str:
        """
        Identify which pipeline stage the error occurred in.

        Args:
            error: The exception that occurred
            retrieval_time: Time spent in retrieval (0 if retrieval didn't complete)
            generation_time: Time spent in generation (0 if generation didn't start)

        Returns:
            String identifying the error stage
        """
        error_msg = str(error).lower()

        # Check for memory-related errors
        if "out of memory" in error_msg or "cuda out of memory" in error_msg:
            return "memory"

        # Check for model loading errors
        if "load" in error_msg and ("model" in error_msg or "tokenizer" in error_msg):
            if generation_time == 0:
                return "generation"
            elif retrieval_time == 0:
                return "retrieval"

        # Determine by timing
        if retrieval_time == 0:
            return "retrieval"
        elif generation_time == 0:
            return "generation"
        else:
            return "general"

    def process_samples(
        self,
        samples: List[Sample],
        use_sequential_loading: bool = True
    ) -> List[PipelineResult]:
        """
        Process multiple samples through the pipeline.

        Args:
            samples: List of samples to process
            use_sequential_loading: Whether to use sequential model loading

        Returns:
            List of pipeline results
        """
        results = []

        logger.info(f"Processing {len(samples)} samples...")

        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i+1}/{len(samples)}: {sample.question_id}")

            result = self.process_query(
                question=sample.question,
                question_id=sample.question_id,
                ground_truth=sample.ground_truth,
                use_sequential_loading=use_sequential_loading
            )

            results.append(result)

            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(samples)} samples")

        logger.info(f"Completed processing {len(samples)} samples")
        return results

    def _update_stats(self, retrieval_time: float, generation_time: float, total_time: float) -> None:
        """Update pipeline performance statistics."""
        self.pipeline_stats["total_queries"] += 1
        self.pipeline_stats["successful_queries"] += 1

        # Running average calculation
        n = self.pipeline_stats["successful_queries"]
        self.pipeline_stats["avg_retrieval_time"] = (
            (self.pipeline_stats["avg_retrieval_time"] * (n-1) + retrieval_time) / n
        )
        self.pipeline_stats["avg_generation_time"] = (
            (self.pipeline_stats["avg_generation_time"] * (n-1) + generation_time) / n
        )
        self.pipeline_stats["avg_total_time"] = (
            (self.pipeline_stats["avg_total_time"] * (n-1) + total_time) / n
        )

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        return {
            **self.pipeline_stats,
            "memory_stats": self.memory_manager.monitor.get_current_stats().__dict__,
            "memory_trend": self.memory_manager.monitor.get_memory_usage_trend()
        }

    # Sprint 6 Enhancement: Advanced monitoring and optimization methods

    def check_performance_triggers(self, retrieval_time: float, generation_time: float,
                                 total_time: float) -> List[str]:
        """
        Check for performance optimization triggers.

        Args:
            retrieval_time: Time taken for retrieval stage
            generation_time: Time taken for generation stage
            total_time: Total pipeline time

        Returns:
            List of triggered optimization actions
        """
        triggers = []

        if retrieval_time > self.performance_monitor["retrieval_time_threshold"]:
            triggers.append("optimize_retrieval")
            logger.warning(f"Retrieval time ({retrieval_time:.2f}s) exceeds threshold")

        if generation_time > self.performance_monitor["generation_time_threshold"]:
            triggers.append("optimize_generation")
            logger.warning(f"Generation time ({generation_time:.2f}s) exceeds threshold")

        if total_time > self.performance_monitor["total_time_threshold"]:
            triggers.append("optimize_pipeline")
            logger.warning(f"Total pipeline time ({total_time:.2f}s) exceeds threshold")

        # Check memory pressure
        memory_stats = self.memory_manager.monitor.get_current_stats()
        if memory_stats.gpu_total_gb > 0:
            memory_utilization = memory_stats.gpu_allocated_gb / memory_stats.gpu_total_gb
            if memory_utilization > self.performance_monitor["memory_optimization_threshold"]:
                triggers.append("optimize_memory")
                logger.warning(f"Memory utilization ({memory_utilization:.1%}) is high")

        return triggers

    def apply_performance_optimizations(self, triggers: List[str]) -> None:
        """
        Apply performance optimizations based on triggers.

        Args:
            triggers: List of optimization triggers to apply
        """
        for trigger in triggers:
            try:
                if trigger == "optimize_memory":
                    self._optimize_memory()
                elif trigger == "optimize_retrieval":
                    self._optimize_retrieval()
                elif trigger == "optimize_generation":
                    self._optimize_generation()
                elif trigger == "optimize_pipeline":
                    self._optimize_pipeline()

                self.pipeline_stats["memory_optimizations_triggered"] += 1
                logger.info(f"Applied optimization: {trigger}")

            except Exception as e:
                logger.error(f"Failed to apply optimization {trigger}: {e}")

    def _optimize_memory(self) -> None:
        """Optimize memory usage when under pressure."""
        logger.info("Applying memory optimization...")

        # Clear unused model components
        if self.retriever_loaded and self.generator_loaded:
            # In sequential mode, we shouldn't have both loaded
            self.unload_retriever()

        # Aggressive memory cleanup
        self.memory_manager.clear_gpu_memory(aggressive=True)

        # Reduce batch sizes if configured
        if hasattr(self.config.retrieval, 'batch_size'):
            self.config.retrieval.batch_size = max(1, self.config.retrieval.batch_size // 2)
            logger.info(f"Reduced retrieval batch size to {self.config.retrieval.batch_size}")

    def _optimize_retrieval(self) -> None:
        """Optimize retrieval performance."""
        logger.info("Applying retrieval optimization...")

        # Reduce top-k if it's too high
        if self.config.retrieval.top_k > 3:
            self.config.retrieval.top_k = max(3, self.config.retrieval.top_k - 1)
            logger.info(f"Reduced top-k to {self.config.retrieval.top_k}")

    def _optimize_generation(self) -> None:
        """Optimize generation performance."""
        logger.info("Applying generation optimization...")

        # Reduce max generation length
        if self.config.generation.max_length > 256:
            self.config.generation.max_length = max(256, self.config.generation.max_length - 128)
            logger.info(f"Reduced max generation length to {self.config.generation.max_length}")

    def _optimize_pipeline(self) -> None:
        """Apply global pipeline optimizations."""
        logger.info("Applying pipeline optimization...")

        # Force sequential loading if not already enabled
        if not hasattr(self, '_force_sequential'):
            self._force_sequential = True
            logger.info("Enabled forced sequential loading")

    def handle_pipeline_error(self, error: Exception, stage: str, question_id: str) -> bool:
        """
        Handle pipeline errors with recovery strategies.

        Args:
            error: The exception that occurred
            stage: Pipeline stage where error occurred
            question_id: Question ID for tracking

        Returns:
            True if recovery was successful, False otherwise
        """
        recovery_key = f"{stage}:{question_id}"
        attempt_count = self.recovery_attempts.get(recovery_key, 0)

        if attempt_count >= self.max_recovery_attempts:
            logger.error(f"Max recovery attempts reached for {recovery_key}")
            return False

        self.recovery_attempts[recovery_key] = attempt_count + 1
        self.pipeline_stats["error_recoveries_performed"] += 1

        logger.warning(f"Attempting recovery for {stage} error (attempt {attempt_count + 1}): {error}")

        try:
            if stage == "retrieval":
                return self._recover_retrieval_error(error)
            elif stage == "generation":
                return self._recover_generation_error(error)
            elif stage == "memory":
                return self._recover_memory_error(error)
            else:
                return self._recover_general_error(error)

        except Exception as recovery_error:
            logger.error(f"Recovery failed: {recovery_error}")
            return False

    def _recover_retrieval_error(self, error: Exception) -> bool:
        """Recover from retrieval errors."""
        # Unload and reload retriever
        if self.retriever_loaded:
            self.unload_retriever()
        self.memory_manager.clear_gpu_memory(aggressive=True)
        time.sleep(1)  # Brief pause
        self.load_retriever()
        return True

    def _recover_generation_error(self, error: Exception) -> bool:
        """Recover from generation errors."""
        # Unload and reload generator
        if self.generator_loaded:
            self.unload_generator()
        self.memory_manager.clear_gpu_memory(aggressive=True)
        time.sleep(1)  # Brief pause
        self.load_generator()
        return True

    def _recover_memory_error(self, error: Exception) -> bool:
        """Recover from memory errors."""
        self.memory_manager.emergency_cleanup()
        # Unload all models
        self.unload_retriever()
        self.unload_generator()
        time.sleep(2)  # Longer pause for memory recovery
        return True

    def _recover_general_error(self, error: Exception) -> bool:
        """General error recovery."""
        self.memory_manager.clear_gpu_memory()
        return True

    def cleanup(self) -> None:
        """Clean up all pipeline resources."""
        logger.info("Cleaning up MRAG pipeline...")

        self.unload_retriever()
        self.unload_generator()
        self.memory_manager.emergency_cleanup()

        logger.info("Pipeline cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()