"""
Show actual generated outputs vs ground truth for debugging.
"""

import logging
from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.dataset import MRAGDataset
from src.evaluation.evaluator import MRAGBenchEvaluator

logging.basicConfig(level=logging.WARNING)  # Suppress most logs
logger = logging.getLogger(__name__)

def show_outputs():
    """Run 3 samples and show actual outputs."""

    print("\n" + "="*80)
    print("SAMPLE OUTPUTS ANALYSIS")
    print("="*80)

    # Load config and data
    config = MRAGConfig.load('config/mrag_bench.yaml')
    dataset = MRAGDataset(config.dataset.data_path)
    evaluator = MRAGBenchEvaluator(config)

    # Get samples from different scenarios
    scenarios = ['angle', 'scope', 'partial']

    # Initialize pipeline
    pipeline = MRAGPipeline(config)

    for scenario in scenarios:
        samples = dataset.get_samples_by_scenario(scenario)
        sample = samples[0]  # First sample from each scenario

        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario.upper()}")
        print(f"{'='*80}")
        print(f"Question ID: {sample.question_id}")
        print(f"Question: {sample.question}")

        # Process query
        result = pipeline.process_query(
            question=sample.question,
            question_id=sample.question_id,
            ground_truth=sample.ground_truth
        )

        print(f"\nGround Truth:     '{sample.ground_truth}'")
        print(f"Generated Answer: '{result.generated_answer}'")

        # Check match
        is_correct = evaluator._is_answer_correct(result.generated_answer, sample.ground_truth)

        # Normalize
        gen_norm = evaluator._normalize_answer(result.generated_answer)
        gt_norm = evaluator._normalize_answer(sample.ground_truth)

        print(f"\nNormalized GT:        '{gt_norm}'")
        print(f"Normalized Generated: '{gen_norm}'")

        # Jaccard similarity
        gen_words = set(gen_norm.split())
        gt_words = set(gt_norm.split())
        if gen_words and gt_words:
            intersection = gen_words.intersection(gt_words)
            union = gen_words.union(gt_words)
            jaccard = len(intersection) / len(union) if union else 0.0
            print(f"\nJaccard Similarity: {jaccard:.3f} (threshold: 0.6)")
            print(f"Intersection: {intersection}")
            print(f"Union: {union}")

        print(f"\nMatch Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        print(f"Retrieval time: {result.retrieval_time:.3f}s")
        print(f"Generation time: {result.generation_time:.3f}s")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    show_outputs()
