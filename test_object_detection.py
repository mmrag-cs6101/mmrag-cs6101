"""
Quick test script for object detection integration.

This script verifies that the object detection module works correctly
without running a full evaluation.
"""

import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_object_detector():
    """Test the ObjectDetector class."""
    logger.info("Testing ObjectDetector...")
    
    try:
        from src.vision.object_detector import ObjectDetector
        
        # Create detector
        detector = ObjectDetector(
            model_name="facebook/detr-resnet-50",
            confidence_threshold=0.7,
            device="cuda",
            max_objects=10
        )
        
        logger.info("✓ ObjectDetector created")
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        logger.info("✓ Dummy image created")
        
        # Load model
        detector.load_model()
        logger.info("✓ DETR model loaded")
        
        # Detect objects
        objects = detector.detect_objects(dummy_image)
        logger.info(f"✓ Detection completed: {len(objects)} objects detected")
        
        # Analyze image
        analysis = detector.analyze_image(dummy_image)
        logger.info(f"✓ Image analysis completed")
        logger.info(f"  Primary objects: {analysis.primary_objects}")
        logger.info(f"  Object counts: {analysis.object_counts}")
        logger.info(f"  Spatial description: {analysis.spatial_description}")
        
        # Test structured text generation
        structured_text = analysis.to_structured_text()
        logger.info(f"✓ Structured text: {structured_text}")
        
        # Cleanup
        detector.unload_model()
        logger.info("✓ Model unloaded")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ObjectDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_pipeline():
    """Test the EnhancedLLaVAPipeline class."""
    logger.info("\nTesting EnhancedLLaVAPipeline...")
    
    try:
        from src.config import MRAGConfig
        from src.generation.llava_enhanced_pipeline import EnhancedLLaVAPipeline
        from src.generation.interface import GenerationConfig, MultimodalContext
        
        # Load config
        config = MRAGConfig.load('config/mrag_bench.yaml')
        logger.info("✓ Config loaded")
        
        # Create generation config
        gen_config = GenerationConfig(
            model_name=config.model.vlm_name,
            device=config.model.device,
            quantization=config.model.quantization,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            do_sample=config.generation.do_sample,
            max_memory_gb=config.performance.memory_limit_gb
        )
        logger.info("✓ Generation config created")
        
        # Create enhanced pipeline
        pipeline = EnhancedLLaVAPipeline(
            config=gen_config,
            use_object_detection=True,
            detection_confidence=0.7,
            max_objects_per_image=10
        )
        logger.info("✓ Enhanced pipeline created")
        
        # Note: We don't load models here to save time
        # Just verify the pipeline can be instantiated
        
        logger.info("✓ Enhanced pipeline test passed (initialization only)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.vision.object_detector import (
            ObjectDetector,
            DetectedObject,
            ImageAnalysis,
            create_enhanced_prompt_with_detection
        )
        logger.info("✓ vision.object_detector imports")
        
        from src.generation.llava_enhanced_pipeline import EnhancedLLaVAPipeline
        logger.info("✓ generation.llava_enhanced_pipeline imports")
        
        from src.generation import EnhancedLLaVAPipeline as EnhancedFromInit
        logger.info("✓ EnhancedLLaVAPipeline available from src.generation")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("OBJECT DETECTION INTEGRATION TEST")
    print("="*80)
    
    results = {}
    
    # Test imports
    results['imports'] = test_imports()
    
    # Test object detector (requires GPU/model download)
    print("\n" + "-"*80)
    response = input("Test ObjectDetector with model loading? (requires GPU, ~160MB download) [y/N]: ")
    if response.lower() == 'y':
        results['object_detector'] = test_object_detector()
    else:
        logger.info("Skipping ObjectDetector test")
        results['object_detector'] = None
    
    # Test enhanced pipeline (initialization only)
    print("\n" + "-"*80)
    results['enhanced_pipeline'] = test_enhanced_pipeline()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    print("="*80)
    
    # Overall result
    failed_tests = [name for name, result in results.items() if result is False]
    if failed_tests:
        print(f"\n✗ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        return False
    else:
        print("\n✓ All tests passed!")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
