"""
LLaVA Generation Pipeline Implementation

Concrete implementation of LLaVA-1.5-7B with 4-bit quantization for multimodal question answering.
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
    AutoProcessor,
    LlavaForConditionalGeneration,
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
    LLaVA-1.5-7B based generation pipeline with configurable quantization.

    Features:
    - Configurable quantization (4-bit, 8-bit, or none) with BitsAndBytes
    - Multimodal prompt construction for visual question answering
    - Dynamic model loading/unloading for memory management
    - Response formatting and validation
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

        # Detect model type for compatibility handling
        self.is_onevision = "onevision" in config.model_name.lower()

        # Quantization configuration for 4-bit
        self.quantization_config = self._create_quantization_config()

        # Generation parameters
        self.generation_kwargs = {
            "max_new_tokens": config.max_length,
            "temperature": config.temperature,
            "do_sample": config.do_sample,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.2,  # Prevent repetitive tokens
            "no_repeat_ngram_size": 3,  # Prevent repeating 3-grams
            "pad_token_id": None,  # Will be set after model loading
            "eos_token_id": None,  # Will be set after model loading
        }

        logger.info("LLaVA generation pipeline initialized")

    def _create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create quantization configuration based on config settings.

        Returns:
            BitsAndBytesConfig for 4-bit or 8-bit, or None for no quantization
        """
        quant_mode = self.config.quantization.lower()

        if quant_mode == "4bit":
            logger.info("Using 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.uint8
            )
        elif quant_mode == "8bit":
            logger.info("Using 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True
            )
        else:
            logger.info("No quantization - loading full precision model")
            return None

    @handle_errors
    def load_model(self) -> None:
        """Load and initialize LLaVA model with configured quantization."""
        if self.model_loaded:
            logger.info("LLaVA model already loaded")
            return

        quant_info = self.config.quantization if self.config.quantization.lower() != "none" else "no quantization"
        logger.info(f"Loading LLaVA-1.5-7B with {quant_info}...")

        with self.memory_manager.memory_guard("LLaVA model loading"):
            try:
                # Load processor first (lighter weight)
                logger.info("Loading LLaVA processor...")
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model_name
                )

                # Load model with optional quantization
                logger.info(f"Loading LLaVA model with {quant_info}...")
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16 if self.config.torch_dtype == "float16" else torch.float32,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True
                }

                # Add quantization config if enabled
                if self.quantization_config is not None:
                    model_kwargs["quantization_config"] = self.quantization_config

                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
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
                # For OneVision: temporarily use only one image
                if self.is_onevision:
                    # Create modified context with single image for OneVision
                    single_image_context = MultimodalContext(
                        question=context.question,
                        images=context.images[:1],  # Only first image
                        choices=context.choices
                    )
                    prompt = self.construct_prompt(single_image_context)
                else:
                    prompt = self.construct_prompt(context)

                # Prepare images for processing
                images = self._prepare_images(context.images)

                if not images:
                    # No valid images, return a default response
                    return GenerationResult(
                        answer="I cannot provide an answer without valid images.",
                        confidence_score=0.0,
                        generation_time=time.time() - start_time,
                        memory_usage=self.get_memory_usage(),
                        metadata={"error": "No valid images provided"}
                    )

                # Use the processor's unified call method
                import torch

                # Ensure images is a list
                image_list = images if isinstance(images, list) else [images]

                # LLaVA-OneVision processes images differently than LLaVA-1.5
                if self.is_onevision:
                    # TEMPORARY: OneVision multi-image support is complex
                    # Use only the first image for now to verify the model works
                    # TODO: Implement proper multi-image support for OneVision
                    logger.warning(f"LLaVA-OneVision: Using only first image of {len(image_list)} images (multi-image not yet supported)")
                    inputs = self.processor(
                        text=prompt,
                        images=image_list[0],  # Single image only
                        return_tensors="pt"
                    )
                else:
                    # LLaVA-1.5: Pass images as flat list [img1, img2, img3]
                    inputs = self.processor(
                        text=prompt,
                        images=image_list,
                        return_tensors="pt"
                    )

                # Move all inputs to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Filter inputs based on model type
                # LLaVA-OneVision doesn't accept batch_num_images or image_sizes
                # LLaVA-1.5 accepts both
                if self.is_onevision:
                    # Remove incompatible parameters for OneVision (uses SiglipVisionModel)
                    excluded_keys = {'batch_num_images', 'image_sizes'}
                    generation_inputs = {k: v for k, v in inputs.items() if k not in excluded_keys}
                    logger.debug("Using LLaVA-OneVision compatible inputs (filtered batch_num_images, image_sizes)")
                else:
                    # Keep all inputs for LLaVA-1.5
                    generation_inputs = inputs
                    logger.debug("Using LLaVA-1.5 compatible inputs")

                # Generate response
                logger.debug(f"Generating response for prompt length: {len(prompt)}")

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **generation_inputs,
                        **self.generation_kwargs
                    )

                # Decode response
                input_token_len = inputs["input_ids"].shape[1]
                response_ids = output_ids[0][input_token_len:]
                response = self.processor.tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True
                ).strip()

                # Post-process response
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
        Construct prompt for multimodal question answering.

        Args:
            context: MultimodalContext with question and images

        Returns:
            Formatted prompt string for the model
        """
        # Add image tokens for each image (required by LLaVA)
        image_tokens = "".join([f"<image>" for _ in context.images])

        # Format the question
        question = context.question.strip()

        # Check if this is a multiple-choice question
        if context.choices and len(context.choices) > 0:
            # MRAG-Bench official prompt format
            # Format: "Instruction: ... {Image}{Image}... Question: ... Choices: (A) ... Answer:"
            choices_text = "\n".join([f"({k}) {v}" for k, v in sorted(context.choices.items())])
            prompt = (
                f"Instruction: You will be given one question concerning several images. "
                f"The first image is the input image, others are retrieved examples to help you. "
                f"Answer with the option's letter from the given choices directly.\n\n"
                f"{image_tokens}\n\n"
                f"Question: {question}\n\n"
                f"Choices:\n{choices_text}\n\n"
                f"Answer:"
            )
        else:
            # Open-ended format (fallback)
            system_prompt = "Answer the question with the most specific term possible. Use underscores between words. No explanation."
            prompt = f"{image_tokens}\n\n{system_prompt}\n\nQuestion: {question}\n\nAnswer:"

        return prompt

    def _extract_answer_choice(self, response: str) -> str:
        """
        Extract the answer choice (A, B, C, D) from model response.

        Args:
            response: Raw model output

        Returns:
            Single letter (A, B, C, or D) or empty string if not found
        """
        import re

        # Remove whitespace
        response = response.strip().upper()

        # Check if response is already just a single letter
        if response in ['A', 'B', 'C', 'D']:
            return response

        # Try to find first occurrence of A, B, C, or D
        # Pattern: Look for standalone letter (not part of word)
        match = re.search(r'\b([ABCD])\b', response)
        if match:
            return match.group(1)

        # Check first character if it's a valid choice
        if response and response[0] in ['A', 'B', 'C', 'D']:
            return response[0]

        # Look for pattern like "Answer: A" or "The answer is B"
        match = re.search(r'(?:answer|choice|select|option)[\s:]*([ABCD])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Default to empty if no valid choice found
        logger.warning(f"Could not extract answer choice from: {response[:100]}")
        return ""

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
        Post-process generated response.

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

        # Response length factor (short, specific answers are better)
        if len(response) > 5:
            confidence += 0.2
        if len(response) > 10:
            confidence += 0.1

        # Penalty for overly long responses (likely verbose/wrong)
        if len(response) > 100:
            confidence -= 0.2

        # Avoid overconfident responses
        if "I cannot" in response or "error" in response.lower():
            confidence = min(confidence, 0.3)

        return max(0.1, min(1.0, confidence))

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