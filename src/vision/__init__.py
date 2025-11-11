"""Vision processing module for MRAG-Bench system."""

from .object_detector import (
    ObjectDetector,
    DetectedObject,
    ImageAnalysis,
    create_enhanced_prompt_with_detection
)

__all__ = [
    "ObjectDetector",
    "DetectedObject",
    "ImageAnalysis",
    "create_enhanced_prompt_with_detection"
]
