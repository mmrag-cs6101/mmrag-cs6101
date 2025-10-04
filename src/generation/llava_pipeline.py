"""
LLaVA Generation Pipeline Implementation

Concrete implementation of LLaVA-1.5-7B with 4-bit quantization for medical image question answering.
Optimized for 16GB VRAM constraints with aggressive memory management.
"""

import os
import gc
import time
import logging
import warnings
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    pipeline
)

from .interface import GenerationPipeline, MultimodalContext, GenerationResult, GenerationConfig
from ..utils.memory_manager import MemoryManager
from ..utils.error_handling import handle_errors, MRAGError

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class LLaVAGenerationPipeline(GenerationPipeline):
    """
    LLaVA-1.5-7B based generation pipeline with 4-bit quantization.

    Features:
    - 4-bit quantization with BitsAndBytes for memory efficiency
    - Multimodal prompt construction for medical domain
    - Dynamic model loading/unloading for memory management
    - Medical-specific response formatting and validation
    - Comprehensive error handling and recovery mechanisms
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize LLaVA generation pipeline.

        Args:
            config: Generation configuration parameters
        """
        super().__init__(config)

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.memory_manager = MemoryManager(
            memory_limit_gb=config.max_memory_gb + 2.0,  # Allow some buffer for LLaVA
            buffer_gb=1.0
        )

        # Model components (loaded on-demand)
        self.model = None
        self.processor = None
        self.model_loaded = False

        # Quantization configuration for 4-bit
        self.quantization_config = self._create_quantization_config()

        # Generation parameters
        self.generation_kwargs = {
            "max_new_tokens": config.max_length,
            "temperature": config.temperature,
            "do_sample": config.do_sample,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "pad_token_id": None,  # Will be set after model loading
            "eos_token_id": None,  # Will be set after model loading
        }

        logger.info("LLaVA generation pipeline initialized")

    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create optimized 4-bit quantization configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8
        )

    @handle_errors
    def load_model(self) -> None:
        """Load and initialize LLaVA model with 4-bit quantization."""
        if self.model_loaded:
            logger.info("LLaVA model already loaded")
            return

        logger.info("Loading LLaVA-1.5-7B with 4-bit quantization...")

        with self.memory_manager.memory_guard("LLaVA model loading"):
            try:
                # Load processor first (lighter weight)
                logger.info("Loading LLaVA processor...")
                self.processor = LlavaNextProcessor.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.config.torch_dtype == "float16" else torch.float32,
                    use_fast=False  # Fast processor doesn't include image_sizes
                )

                # Load model with quantization
                logger.info("Loading LLaVA model with 4-bit quantization...")
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    quantization_config=self.quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.config.torch_dtype == "float16" else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

                # Set generation parameters with proper token IDs
                if self.processor.tokenizer.pad_token_id is not None:
                    self.generation_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id
                if self.processor.tokenizer.eos_token_id is not None:
                    self.generation_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id

                # Set model to evaluation mode
                self.model.eval()

                # Clear any intermediate memory
                self.memory_manager.clear_gpu_memory()

                self.model_loaded = True
                memory_stats = self.memory_manager.monitor.log_memory_stats("After LLaVA loading")

                logger.info(
                    f"LLaVA model loaded successfully. "
                    f"GPU memory: {memory_stats.gpu_allocated_gb:.2f}GB allocated"
                )

            except Exception as e:
                self.model_loaded = False
                self.memory_manager.emergency_cleanup()
                raise MRAGError(f"Failed to load LLaVA model: {str(e)}") from e

    @handle_errors
    def generate_answer(self, context: MultimodalContext) -> GenerationResult:
        """
        Generate answer for multimodal context using LLaVA.

        Args:
            context: MultimodalContext with question and retrieved images

        Returns:
            GenerationResult with generated answer and metadata
        """
        if not self.model_loaded:
            self.load_model()

        start_time = time.time()
        initial_memory = self.memory_manager.monitor.get_current_stats()

        with self.memory_manager.memory_guard("LLaVA generation"):
            try:
                # Construct multimodal prompt
                prompt = self.construct_prompt(context)

                # Prepare images for processing
                images = self._prepare_images(context.images)

                if not images:
                    # No valid images, return a default response
                    return GenerationResult(
                        answer="I cannot provide an answer without valid medical images.",
                        confidence_score=0.0,
                        generation_time=time.time() - start_time,
                        memory_usage=self.get_memory_usage(),
                        metadata={"error": "No valid images provided"}
                    )

                # Process text and images separately to handle image_sizes properly
                import torch

                # Ensure images is a list
                image_list = images if isinstance(images, list) else [images]

                # Process images separately to get pixel_values
                image_inputs = self.processor.image_processor(
                    images=image_list,
                    return_tensors="pt"
                )

                # Process text
                text_inputs = self.processor.tokenizer(
                    text=prompt,
                    return_tensors="pt"
                )

                # Manually add image_sizes (required by LLaVA Next)
                image_sizes = torch.tensor([[img.size[1], img.size[0]] for img in image_list])

                # Combine inputs
                inputs = {
                    **text_inputs,
                    **image_inputs,
                    "image_sizes": image_sizes
                }

                # Move all inputs to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Generate response
                logger.debug(f"Generating response for prompt length: {len(prompt)}")

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        **self.generation_kwargs
                    )

                # Decode response
                input_token_len = inputs["input_ids"].shape[1]
                response_ids = output_ids[0][input_token_len:]
                response = self.processor.tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True
                ).strip()

                # Post-process response for medical domain
                response = self._post_process_response(response)

                # Calculate generation time and memory usage
                generation_time = time.time() - start_time
                final_memory = self.memory_manager.monitor.get_current_stats()
                memory_usage = {
                    "initial_allocated_gb": initial_memory.gpu_allocated_gb,
                    "peak_allocated_gb": final_memory.gpu_allocated_gb,
                    "memory_increase_gb": final_memory.gpu_allocated_gb - initial_memory.gpu_allocated_gb
                }

                # Calculate confidence score (simple heuristic based on response length and content)
                confidence_score = self._calculate_confidence_score(response, context)

                logger.info(f"Generated response in {generation_time:.2f}s")

                return GenerationResult(
                    answer=response,
                    confidence_score=confidence_score,
                    generation_time=generation_time,
                    memory_usage=memory_usage,
                    metadata={
                        "prompt_length": len(prompt),
                        "response_length": len(response),
                        "num_images": len(images)
                    }
                )

            except Exception as e:
                self.memory_manager.emergency_cleanup()
                import traceback
                logger.error(f"Generation failed: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

                return GenerationResult(
                    answer="I apologize, but I encountered an error processing your question.",
                    confidence_score=0.0,
                    generation_time=time.time() - start_time,
                    memory_usage=self.get_memory_usage(),
                    metadata={"error": str(e)}
                )

    def construct_prompt(self, context: MultimodalContext) -> str:
        """
        Construct optimized prompt for medical image question answering.

        Args:
            context: MultimodalContext with question and images

        Returns:
            Formatted prompt string optimized for medical domain
        """
        # Medical domain-specific prompt template
        system_prompt = (
            "You are a medical AI assistant specialized in analyzing medical images. "
            "Carefully examine the provided medical images and answer the question accurately. "
            "Focus on observable medical findings and provide clear, concise responses."
        )

        # Format the question
        question = context.question.strip()
        if not question.endswith(('?', '.', '!')):
            question += "?"

        # Handle retrieved images context
        if len(context.images) > 1:
            image_context = f"Based on the {len(context.images)} medical images provided, "
        elif len(context.images) == 1:
            image_context = "Based on the medical image provided, "
        else:
            image_context = ""

        # Construct final prompt
        prompt = f"{system_prompt}\n\n{image_context}{question}\n\nAnswer:"

        return prompt

    def _prepare_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Prepare and validate images for LLaVA processing.

        Args:
            images: List of PIL Images

        Returns:
            List of validated and preprocessed images
        """
        valid_images = []

        for i, image in enumerate(images):
            try:
                if image is None:
                    logger.warning(f"Image {i} is None, skipping")
                    continue

                # Ensure image is in RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Validate image size
                if image.size[0] < 32 or image.size[1] < 32:
                    logger.warning(f"Image {i} too small ({image.size}), skipping")
                    continue

                # Resize if too large (memory optimization)
                max_size = 512
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)

                valid_images.append(image)

            except Exception as e:
                logger.warning(f"Error processing image {i}: {e}")
                continue

        logger.debug(f"Prepared {len(valid_images)} valid images from {len(images)} input images")
        return valid_images

    def _post_process_response(self, response: str) -> str:
        """
        Post-process generated response for medical domain.

        Args:
            response: Raw model response

        Returns:
            Cleaned and formatted response
        """
        # Remove common artifacts
        response = response.strip()

        # Remove repeated phrases
        lines = response.split('\n')
        unique_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in unique_lines:
                unique_lines.append(line)

        response = '\n'.join(unique_lines)

        # Ensure response starts with capital letter
        if response and response[0].islower():
            response = response[0].upper() + response[1:]

        # Ensure response ends with proper punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += '.'

        return response

    def _calculate_confidence_score(self, response: str, context: MultimodalContext) -> float:
        """
        Calculate confidence score for generated response.

        Args:
            response: Generated response
            context: Original multimodal context

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple heuristic-based confidence calculation
        confidence = 0.5  # Base confidence

        # Response length factor
        if len(response) > 20:
            confidence += 0.2
        if len(response) > 50:
            confidence += 0.1

        # Medical keywords factor
        medical_keywords = [
            'medical', 'clinical', 'diagnosis', 'symptom', 'treatment',
            'patient', 'condition', 'anatomical', 'pathology', 'imaging'
        ]
        keyword_count = sum(1 for keyword in medical_keywords if keyword.lower() in response.lower())
        confidence += min(0.3, keyword_count * 0.1)

        # Avoid overconfident responses
        if "I cannot" in response or "error" in response.lower():
            confidence = min(confidence, 0.3)

        return min(1.0, confidence)

    @handle_errors
    def clear_memory(self) -> None:
        """Clear GPU memory and release model resources."""
        logger.debug("Clearing LLaVA generation memory")

        # Clear any cached tensors
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()

        # Standard memory cleanup
        self.memory_manager.clear_gpu_memory(aggressive=True)

        logger.debug("LLaVA memory cleared")

    @handle_errors
    def unload_model(self) -> None:
        """Unload model from memory for dynamic loading."""
        if not self.model_loaded:
            logger.debug("LLaVA model not loaded, nothing to unload")
            return

        logger.info("Unloading LLaVA model...")

        try:
            # Move model to CPU and delete
            if self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None

            if self.processor is not None:
                del self.processor
                self.processor = None

            # Clear memory
            self.memory_manager.emergency_cleanup()

            self.model_loaded = False
            logger.info("LLaVA model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading LLaVA model: {e}")
            self.memory_manager.emergency_cleanup()
            raise

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage in GB
        """
        return self.memory_manager.monitor.get_current_stats().__dict__

    def validate_memory_constraints(self) -> bool:
        """
        Check if current memory usage is within configured limits.

        Returns:
            True if within limits, False otherwise
        """
        current_stats = self.memory_manager.monitor.get_current_stats()
        return current_stats.is_within_limit(self.config.max_memory_gb)