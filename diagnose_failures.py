"""
Diagnostic script to identify why 98.5% of queries are failing.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.dataset import MRAGDataset

def test_single_query():
    """Test a single query end-to-end with detailed logging."""
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC TEST: Single Query End-to-End")
    logger.info("=" * 80)

    # Load config
    config = MRAGConfig.load('config/mrag_bench.yaml')

    # Load dataset to get a real sample
    dataset = MRAGDataset(config.dataset.data_path)
    samples = dataset.get_samples_by_scenario('angle')
    sample = samples[0]  # First angle sample

    logger.info(f"\nTest Sample:")
    logger.info(f"  Question ID: {sample.question_id}")
    logger.info(f"  Question: {sample.question}")
    logger.info(f"  Ground Truth: {sample.ground_truth}")
    logger.info(f"  Scenario: {sample.scenario}")

    # Initialize pipeline
    logger.info("\nInitializing pipeline...")
    pipeline = MRAGPipeline(config)

    # Process query
    logger.info("\nProcessing query...")
    try:
        result = pipeline.process_query(
            question=sample.question,
            question_id=sample.question_id,
            ground_truth=sample.ground_truth
        )

        logger.info("\n" + "=" * 80)
        logger.info("RESULT:")
        logger.info("=" * 80)
        if result and hasattr(result, 'answer'):
            logger.info(f"✅ SUCCESS")
            logger.info(f"  Generated Answer: {result.answer}")
            logger.info(f"  Ground Truth: {sample.ground_truth}")
            logger.info(f"  Correct: {result.answer.lower().strip() == sample.ground_truth.lower().strip()}")
            logger.info(f"  Retrieval Time: {result.retrieval_time:.3f}s")
            logger.info(f"  Generation Time: {result.generation_time:.3f}s")
            logger.info(f"  Total Time: {result.total_time:.3f}s")
        else:
            logger.error(f"❌ FAILED - Result is None or missing answer attribute")
            logger.error(f"  Result type: {type(result)}")
            logger.error(f"  Result value: {result}")

    except Exception as e:
        logger.error(f"\n❌ EXCEPTION during query processing:")
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Error message: {str(e)}")
        import traceback
        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        return False

    return True

def test_retrieval():
    """Test retrieval component in isolation."""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC TEST: Retrieval Component")
    logger.info("=" * 80)

    config = MRAGConfig.load('config/mrag_bench.yaml')
    pipeline = MRAGPipeline(config)

    test_question = "Can you identify this animal?"
    logger.info(f"\nTest Question: {test_question}")

    try:
        # Encode query
        logger.info("Encoding query...")
        query_embedding = pipeline.retriever.encode_text(test_question)
        logger.info(f"  Query embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'N/A'}")

        # Search
        logger.info("Searching for similar images...")
        results = pipeline.retriever.search(query_embedding, top_k=5)

        logger.info(f"\n✅ Retrieved {len(results)} results:")
        for idx, (image_path, score) in enumerate(results):
            logger.info(f"  {idx+1}. {Path(image_path).name} (score: {score:.4f})")
            # Check if file exists
            if Path(image_path).exists():
                logger.info(f"      ✅ File exists")
            else:
                logger.error(f"      ❌ File NOT FOUND")

    except Exception as e:
        logger.error(f"\n❌ Retrieval failed:")
        logger.error(f"  Error: {str(e)}")
        import traceback
        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        return False

    return True

def test_generation():
    """Test generation component in isolation."""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC TEST: Generation Component")
    logger.info("=" * 80)

    from src.generation import LLaVAGenerationPipeline, MultimodalContext, GenerationConfig
    from PIL import Image

    config = MRAGConfig.load('config/mrag_bench.yaml')

    # Create generation config
    gen_config = GenerationConfig(
        model_name=config.model.vlm_name,
        max_length=config.model.max_length,
        temperature=config.model.temperature,
        do_sample=config.model.do_sample,
        top_p=config.model.top_p,
        top_k=config.model.top_k,
        max_memory_gb=config.model.max_memory_gb,
        device=config.model.device,
        torch_dtype=config.model.torch_dtype
    )

    logger.info("Initializing generation pipeline...")
    gen_pipeline = LLaVAGenerationPipeline(gen_config)

    logger.info("Loading model...")
    gen_pipeline.load_model()

    # Find a test image
    image_dir = Path("data/mrag_bench/images")
    image_files = list(image_dir.glob("image_*.jpg"))
    if not image_files:
        logger.error("❌ No test images found")
        return False

    test_image_path = image_files[0]
    logger.info(f"\nTest image: {test_image_path.name}")

    try:
        # Load image
        image = Image.open(test_image_path).convert('RGB')
        logger.info(f"  Image size: {image.size}")

        # Create context
        context = MultimodalContext(
            question="What do you see in this image?",
            images=[image]
        )

        # Generate
        logger.info("\nGenerating answer...")
        result = gen_pipeline.generate_answer(context)

        logger.info(f"\n✅ Generation successful:")
        logger.info(f"  Answer: {result.answer}")
        logger.info(f"  Confidence: {result.confidence_score:.3f}")
        logger.info(f"  Generation time: {result.generation_time:.3f}s")

    except Exception as e:
        logger.error(f"\n❌ Generation failed:")
        logger.error(f"  Error: {str(e)}")
        import traceback
        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        return False

    return True

if __name__ == "__main__":
    logger.info("MRAG-Bench Failure Diagnostics")
    logger.info("=" * 80)

    # Run tests
    tests = [
        ("Retrieval Component", test_retrieval),
        ("Generation Component", test_generation),
        ("End-to-End Query", test_single_query),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")

    # Exit code
    sys.exit(0 if all(results.values()) else 1)
