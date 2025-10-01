"""
End-to-End MRAG Pipeline

Orchestrates retrieval and generation for complete multimodal RAG system.
Integrates CLIP retrieval with LLaVA generation with memory management.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image

from .config import MRAGConfig
from .dataset import MRAGDataset, Sample
from .retrieval import CLIPRetriever, RetrievalConfig
from .generation import LLaVAGenerationPipeline, GenerationConfig, MultimodalContext, GenerationResult
from .utils.memory_manager import MemoryManager
from .utils.error_handling import handle_errors, MRAGError


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
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_total_time": 0.0
        }

        logger.info("MRAG pipeline initialized")

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
                device=self.config.model.device,
                cache_embeddings=self.config.dataset.cache_embeddings,
                embedding_cache_path=self.config.dataset.embedding_cache_path
            )

            self.retriever = CLIPRetriever(retrieval_config)
            self.retriever.load_model()

            # Build index if not cached
            if self.dataset is None:
                self.initialize_dataset()

            corpus_paths = self.dataset.get_retrieval_corpus()
            self.retriever.build_index(corpus_paths)

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

            # Update statistics
            self._update_stats(retrieval_time, generation_time, total_time)

            # Create pipeline result
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
                }
            )

            return result

        except Exception as e:
            logger.error(f"Pipeline processing failed for question {question_id}: {e}")
            self.memory_manager.emergency_cleanup()

            # Return error result
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
                metadata={"error": str(e)}
            )

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