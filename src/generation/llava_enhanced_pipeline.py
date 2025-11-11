"""
Enhanced LLaVA Generation Pipeline with Object Detection

This module extends the standard LLaVA pipeline by integrating object detection
to provide structured visual reasoning capabilities.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from PIL import Image

from .llava_pipeline import LLaVAGenerationPipeline
from .interface import MultimodalContext, GenerationResult, GenerationConfig
from ..vision.object_detector import ObjectDetector, create_enhanced_prompt_with_detection
from ..utils.error_handling import handle_errors

logger = logging.getLogger(__name__)


class EnhancedLLaVAPipeline(LLaVAGenerationPipeline):
    """
    Enhanced LLaVA pipeline with integrated object detection.
    
    This pipeline:
    1. Analyzes images using object detection (DETR)
    2. Converts visual content to structured text
    3. Enhances prompts with object detection information
    4. Improves visual reasoning accuracy through explicit visual grounding
    """
    
    def __init__(
        self,
        config: GenerationConfig,
        use_object_detection: bool = True,
        detection_confidence: float = 0.7,
        max_objects_per_image: int = 10
    ):
        """
        Initialize enhanced LLaVA pipeline.
        
        Args:
            config: Generation configuration
            use_object_detection: Enable/disable object detection enhancement
            detection_confidence: Confidence threshold for object detection
            max_objects_per_image: Maximum objects to detect per image
        """
        super().__init__(config)
        
        self.use_object_detection = use_object_detection
        self.object_detector = None
        
        if self.use_object_detection:
            self.object_detector = ObjectDetector(
                model_name="facebook/detr-resnet-50",
                confidence_threshold=detection_confidence,
                device=config.device,
                max_objects=max_objects_per_image
            )
            logger.info("Enhanced LLaVA pipeline initialized with object detection")
        else:
            logger.info("Enhanced LLaVA pipeline initialized without object detection")
    
    @handle_errors
    def load_model(self) -> None:
        """Load both LLaVA and object detection models."""
        # Load base LLaVA model
        super().load_model()
        
        # Load object detection model if enabled
        if self.use_object_detection and self.object_detector is not None:
            logger.info("Loading object detection model...")
            self.object_detector.load_model()
            logger.info("Object detection model loaded")
    
    @handle_errors
    def generate_answer(self, context: MultimodalContext) -> GenerationResult:
        """
        Generate answer with enhanced visual reasoning using object detection.
        
        Args:
            context: MultimodalContext with question and images
            
        Returns:
            GenerationResult with enhanced visual reasoning
        """
        if not self.model_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # Perform object detection if enabled
        enhanced_context = context
        detection_metadata = {}
        
        if self.use_object_detection and self.object_detector is not None:
            try:
                logger.debug("Performing object detection on images...")
                detection_start = time.time()
                
                # Analyze all images
                image_analyses = self.object_detector.analyze_multiple_images(context.images)
                detection_time = time.time() - detection_start
                
                # Log detected objects
                total_objects = sum(len(analysis.objects) for analysis in image_analyses)
                logger.info(f"Detected {total_objects} objects across {len(context.images)} images in {detection_time:.2f}s")
                
                # Create enhanced context with object detection information
                enhanced_context = self._create_enhanced_context(context, image_analyses)
                
                # Store detection metadata
                detection_metadata = {
                    "detection_enabled": True,
                    "detection_time": detection_time,
                    "total_objects_detected": total_objects,
                    "objects_per_image": [len(analysis.objects) for analysis in image_analyses],
                    "primary_objects": [analysis.primary_objects for analysis in image_analyses]
                }
                
            except Exception as e:
                logger.warning(f"Object detection failed, falling back to standard pipeline: {e}")
                detection_metadata = {"detection_enabled": False, "error": str(e)}
        else:
            detection_metadata = {"detection_enabled": False}
        
        # Generate answer using enhanced context
        result = super().generate_answer(enhanced_context)
        
        # Add detection metadata to result
        if result.metadata is None:
            result.metadata = {}
        result.metadata.update(detection_metadata)
        
        return result
    
    def _create_enhanced_context(
        self,
        original_context: MultimodalContext,
        image_analyses: List
    ) -> MultimodalContext:
        """
        Create enhanced context with object detection information.
        
        Args:
            original_context: Original multimodal context
            image_analyses: List of ImageAnalysis from object detection
            
        Returns:
            Enhanced MultimodalContext with enriched prompt
        """
        # Build visual context description
        visual_descriptions = []
        for i, analysis in enumerate(image_analyses, 1):
            if analysis.objects:
                visual_descriptions.append(
                    f"Image {i} contains: {analysis.to_structured_text()}"
                )
        
        # Create enhanced question with visual context
        if visual_descriptions:
            visual_context = "\n".join(visual_descriptions)
            enhanced_question = f"""Visual Analysis:
{visual_context}

Original Question: {original_context.question}

Based on the detected objects and visual content above, answer the question."""
        else:
            enhanced_question = original_context.question
        
        # Create new context with enhanced question
        return MultimodalContext(
            question=enhanced_question,
            images=original_context.images,
            choices=original_context.choices,
            retrieved_context=original_context.retrieved_context,
            metadata=original_context.metadata
        )
    
    def construct_prompt(self, context: MultimodalContext) -> str:
        """
        Construct enhanced prompt with object detection information.
        
        This overrides the base method to include visual analysis in the prompt.
        
        Args:
            context: Enhanced MultimodalContext
            
        Returns:
            Formatted prompt with visual reasoning guidance
        """
        # Add image tokens for each image
        image_tokens = "".join([f"<image>" for _ in context.images])
        
        # Format the question (already enhanced with visual analysis)
        question = context.question.strip()
        
        # Check if this is a multiple-choice question
        if context.choices and len(context.choices) > 0:
            # Enhanced MRAG-Bench format with visual reasoning emphasis
            choices_text = "\n".join([f"({k}) {v}" for k, v in sorted(context.choices.items())])
            prompt = (
                f"Instruction: You will be given one question concerning several images. "
                f"The first image is the input image, others are retrieved examples to help you. "
                f"Use the visual analysis provided to inform your reasoning. "
                f"Answer with the option's letter from the given choices directly.\n\n"
                f"{image_tokens}\n\n"
                f"{question}\n\n"
                f"Choices:\n{choices_text}\n\n"
                f"Answer:"
            )
        else:
            # Open-ended format with visual reasoning
            system_prompt = (
                "Use the detected objects and visual content to answer precisely. "
                "Answer with the most specific term possible. Use underscores between words. No explanation."
            )
            prompt = f"{image_tokens}\n\n{system_prompt}\n\n{question}\n\nAnswer:"
        
        return prompt
    
    @handle_errors
    def unload_model(self) -> None:
        """Unload both LLaVA and object detection models."""
        # Unload object detection model first
        if self.object_detector is not None:
            try:
                self.object_detector.unload_model()
            except Exception as e:
                logger.warning(f"Error unloading object detector: {e}")
        
        # Unload base LLaVA model
        super().unload_model()
        
        logger.info("Enhanced LLaVA pipeline unloaded")
    
    def toggle_object_detection(self, enabled: bool) -> None:
        """
        Enable or disable object detection at runtime.
        
        Args:
            enabled: Whether to use object detection
        """
        self.use_object_detection = enabled
        logger.info(f"Object detection {'enabled' if enabled else 'disabled'}")
