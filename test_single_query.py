"""
Test a single query to see actual generated output vs ground truth.
"""

import logging
from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.dataset import MRAGDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query():
    """Test a single query and show outputs."""

    # Load config and data
    config = MRAGConfig.load('config/mrag_bench.yaml')
    dataset = MRAGDataset(config.dataset.data_path)

    # Get a sample from angle scenario
    angle_samples = dataset.get_samples_by_scenario('angle')
    sample = angle_samples[0]

    print("\n" + "="*80)
    print("SINGLE QUERY TEST")
    print("="*80)
    print(f"\nQuestion ID: {sample.question_id}")
    print(f"Question: {sample.question}")
    print(f"Ground Truth: '{sample.ground_truth}'")

    # Initialize pipeline
    pipeline = MRAGPipeline(config)

    # Process query
    print("\nProcessing query...")
    result = pipeline.process_query(
        question=sample.question,
        question_id=sample.question_id,
        ground_truth=sample.ground_truth
    )

    # Show results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nGenerated Answer: '{result.generated_answer}'")
    print(f"Ground Truth:     '{sample.ground_truth}'")

    # Normalize and compare
    from src.evaluation.evaluator import MRAGBenchEvaluator
    evaluator = MRAGBenchEvaluator(config)

    gen_norm = evaluator._normalize_answer(result.generated_answer)
    gt_norm = evaluator._normalize_answer(sample.ground_truth)

    print(f"\nNormalized Generated: '{gen_norm}'")
    print(f"Normalized GT:        '{gt_norm}'")

    # Check match
    is_correct = evaluator._is_answer_correct(result.generated_answer, sample.ground_truth)
    print(f"\nMatch Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

    # Calculate similarity
    gen_words = set(gen_norm.split())
    gt_words = set(gt_norm.split())
    intersection = gen_words.intersection(gt_words)
    union = gen_words.union(gt_words)
    jaccard = len(intersection) / len(union) if union else 0.0

    print(f"\nSimilarity Metrics:")
    print(f"  Generated words: {gen_words}")
    print(f"  GT words: {gt_words}")
    print(f"  Intersection: {intersection}")
    print(f"  Jaccard similarity: {jaccard:.3f} (threshold: 0.6)")

    # Show retrieval info
    print(f"\nRetrieval Info:")
    print(f"  Retrieved {len(result.retrieved_images)} images")
    for i, (img_path, score) in enumerate(zip(result.retrieved_images, result.retrieval_scores)):
        print(f"    {i+1}. {img_path.split('/')[-1]} (score: {score:.3f})")

    # Timing
    print(f"\nTiming:")
    print(f"  Retrieval: {result.retrieval_time:.3f}s")
    print(f"  Generation: {result.generation_time:.3f}s")
    print(f"  Total: {result.total_time:.3f}s")

    print("\n" + "="*80)

if __name__ == "__main__":
    test_query()
