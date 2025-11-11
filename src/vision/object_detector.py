"""
Object Detection Module for Visual Reasoning Enhancement

This module uses DETR (DEtection TRansformer) to extract structured information
from images, converting visual content into text descriptions that enhance
the AI's visual reasoning capabilities.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Represents a detected object in an image."""
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    
    def __str__(self) -> str:
        return f"{self.label} (conf: {self.confidence:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox
        }


@dataclass
class ImageAnalysis:
    """Structured analysis of an image."""
    objects: List[DetectedObject]
    object_counts: Dict[str, int]
    primary_objects: List[str]  # Top objects by confidence
    spatial_description: str
    
    def to_structured_text(self) -> str:
        """Convert analysis to structured text for prompt enhancement."""
        if not self.objects:
            return "No objects detected in this image."
        
        # Build structured description
        parts = []
        
        # Primary objects
        if self.primary_objects:
            parts.append(f"Main objects: {', '.join(self.primary_objects)}")
        
        # Object counts
        if self.object_counts:
            count_str = ", ".join([f"{count} {obj}" for obj, count in self.object_counts.items()])
            parts.append(f"Detected: {count_str}")
        
        # Spatial information
        if self.spatial_description:
            parts.append(f"Layout: {self.spatial_description}")
        
        return ". ".join(parts) + "."


class ObjectDetector:
    """
    Object detection pipeline using DETR for visual content extraction.
    
    Features:
    - Detects objects with bounding boxes and confidence scores
    - Converts visual content to structured text descriptions
    - Provides spatial reasoning information
    - Optimized for memory efficiency
    """
    
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        confidence_threshold: float = 0.7,
        device: str = "cuda",
        max_objects: int = 10
    ):
        """
        Initialize object detector.
        
        Args:
            model_name: HuggingFace model name for DETR
            confidence_threshold: Minimum confidence for object detection
            device: Device to run model on (cuda/cpu)
            max_objects: Maximum number of objects to return per image
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_objects = max_objects
        
        self.processor = None
        self.model = None
        self.model_loaded = False
        
        logger.info(f"ObjectDetector initialized with {model_name}")
    
    def load_model(self) -> None:
        """Load DETR model and processor."""
        if self.model_loaded:
            logger.info("Object detection model already loaded")
            return
        
        logger.info(f"Loading object detection model: {self.model_name}")
        
        try:
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info("Object detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load object detection model: {e}")
            raise
    
    def detect_objects(self, image: Image.Image) -> List[DetectedObject]:
        """
        Detect objects in an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            List of detected objects with labels, confidence, and bounding boxes
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )[0]
            
            # Extract detected objects
            detected_objects = []
            for score, label, box in zip(
                results["scores"],
                results["labels"],
                results["boxes"]
            ):
                if len(detected_objects) >= self.max_objects:
                    break
                
                label_name = self.model.config.id2label[label.item()]
                confidence = score.item()
                bbox = tuple(box.tolist())
                
                detected_objects.append(DetectedObject(
                    label=label_name,
                    confidence=confidence,
                    bbox=bbox
                ))
            
            logger.debug(f"Detected {len(detected_objects)} objects in image")
            return detected_objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def analyze_image(self, image: Image.Image) -> ImageAnalysis:
        """
        Perform comprehensive analysis of an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            ImageAnalysis with structured information
        """
        objects = self.detect_objects(image)
        
        if not objects:
            return ImageAnalysis(
                objects=[],
                object_counts={},
                primary_objects=[],
                spatial_description=""
            )
        
        # Count objects by label
        object_counts = {}
        for obj in objects:
            object_counts[obj.label] = object_counts.get(obj.label, 0) + 1
        
        # Get primary objects (top 3 by confidence)
        sorted_objects = sorted(objects, key=lambda x: x.confidence, reverse=True)
        primary_objects = [obj.label for obj in sorted_objects[:3]]
        
        # Generate spatial description
        spatial_description = self._generate_spatial_description(objects, image.size)
        
        return ImageAnalysis(
            objects=objects,
            object_counts=object_counts,
            primary_objects=primary_objects,
            spatial_description=spatial_description
        )
    
    def _generate_spatial_description(
        self,
        objects: List[DetectedObject],
        image_size: Tuple[int, int]
    ) -> str:
        """
        Generate spatial description of object layout.
        
        Args:
            objects: List of detected objects
            image_size: (width, height) of image
            
        Returns:
            Natural language spatial description
        """
        if not objects:
            return ""
        
        width, height = image_size
        
        # Categorize objects by position
        top_objects = []
        center_objects = []
        bottom_objects = []
        
        for obj in objects:
            x_min, y_min, x_max, y_max = obj.bbox
            center_y = (y_min + y_max) / 2
            
            if center_y < height * 0.33:
                top_objects.append(obj.label)
            elif center_y < height * 0.67:
                center_objects.append(obj.label)
            else:
                bottom_objects.append(obj.label)
        
        # Build description
        parts = []
        if top_objects:
            parts.append(f"{', '.join(top_objects[:2])} in upper area")
        if center_objects:
            parts.append(f"{', '.join(center_objects[:2])} in center")
        if bottom_objects:
            parts.append(f"{', '.join(bottom_objects[:2])} in lower area")
        
        return ", ".join(parts) if parts else "objects distributed across image"
    
    def analyze_multiple_images(self, images: List[Image.Image]) -> List[ImageAnalysis]:
        """
        Analyze multiple images in batch.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of ImageAnalysis results
        """
        return [self.analyze_image(img) for img in images]
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if not self.model_loaded:
            return
        
        logger.info("Unloading object detection model")
        
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        torch.cuda.empty_cache()
        self.model_loaded = False
        
        logger.info("Object detection model unloaded")


def create_enhanced_prompt_with_detection(
    question: str,
    choices: Dict[str, str],
    image_analyses: List[ImageAnalysis]
) -> str:
    """
    Create an enhanced prompt that includes object detection information.
    
    Args:
        question: Original question
        choices: Multiple choice options
        image_analyses: List of ImageAnalysis for each image
        
    Returns:
        Enhanced prompt with structured visual information
    """
    # Build visual context from object detection
    visual_context_parts = []
    
    for i, analysis in enumerate(image_analyses, 1):
        if analysis.objects:
            visual_context_parts.append(
                f"Image {i}: {analysis.to_structured_text()}"
            )
    
    visual_context = "\n".join(visual_context_parts) if visual_context_parts else "Visual analysis not available."
    
    # Format choices
    choices_text = "\n".join([f"({k}) {v}" for k, v in sorted(choices.items())])
    
    # Construct enhanced prompt
    enhanced_prompt = f"""Visual Content Analysis:
{visual_context}

Based on the visual analysis above and the images provided, answer the following question:

Question: {question}

Choices:
{choices_text}

Instructions:
1. Consider the detected objects and their spatial relationships
2. Compare the visual content across all images
3. Use the structured analysis to inform your reasoning
4. Answer with only the letter (A, B, C, or D) of the correct choice

Answer:"""
    
    return enhanced_prompt
