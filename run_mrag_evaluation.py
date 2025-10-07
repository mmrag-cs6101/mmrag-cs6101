"""
Run MRAG-Bench evaluation using official HuggingFace dataset with multiple-choice format.
"""

import logging
import time
from pathlib import Path
from datasets import load_dataset
from PIL import Image

from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.generation.interface import MultimodalContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_answer_choice(response: str) -> str:
    """Extract answer choice (A, B, C, D) from model response."""
    import re

    response = response.strip().upper()

    # Check if response is already just a single letter
    if response in ['A', 'B', 'C', 'D']:
        return response

    # Try to find first occurrence of A, B, C, or D
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)

    # Check first character if it's a valid choice
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]

    return ""


def evaluate_mrag_bench(max_samples_per_scenario: int = 10, use_gt_images: bool = True):
    """
    Evaluate on MRAG-Bench using official HuggingFace dataset.

    Args:
        max_samples_per_scenario: Max samples to test per scenario
        use_gt_images: If True, use ground-truth images; if False, use retrieved images
    """
    # Load configuration
    config = MRAGConfig.load('config/mrag_bench.yaml')

    # Initialize pipeline
    pipeline = MRAGPipeline(config)
    pipeline.load_generator()  # Only need generator for multiple-choice

    # Load official dataset
    logger.info("Loading MRAG-Bench from HuggingFace...")
    dataset = load_dataset("uclanlp/MRAG-Bench", split="test")
    logger.info(f"Loaded {len(dataset)} samples")

    # Scenario mapping (MRAG-Bench categories -> perspective change types)
    scenario_mapping = {
        'Scope': 'scope',
        'Obstruction': 'occlusion',  # Fixed: Obstruction maps to occlusion, not scope
        'Temporal': 'partial',
        'Deformation': 'scope',
        'Biological': 'angle',
        'Angle': 'angle',
        'Partial': 'partial',
        'Incomplete': 'partial',
        'Others': 'scope'
    }

    # Collect samples by scenario
    scenario_samples = {'angle': [], 'partial': [], 'scope': [], 'occlusion': []}

    for idx, sample in enumerate(dataset):
        scenario = sample['scenario']
        perspective_type = scenario_mapping.get(scenario, 'scope')

        if perspective_type in scenario_samples:
            scenario_samples[perspective_type].append((idx, sample))

    # Evaluate each scenario
    results = {}

    for scenario_name in ['angle', 'partial', 'scope', 'occlusion']:
        samples = scenario_samples[scenario_name][:max_samples_per_scenario]
        logger.info(f"\nEvaluating {scenario_name} scenario ({len(samples)} samples)...")

        correct = 0
        total = len(samples)

        for i, (idx, sample) in enumerate(samples):
            # Prepare choices
            choices = {
                "A": sample["A"],
                "B": sample["B"],
                "C": sample["C"],
                "D": sample["D"]
            }

            # Use GT images or retrieved images
            images = sample["gt_images"] if use_gt_images else sample["retrieved_images"]

            # Limit to 3 images (LLaVA constraint)
            images = images[:3]

            # Create context
            context = MultimodalContext(
                question=sample["question"],
                images=images,
                choices=choices
            )

            # Generate answer
            try:
                result = pipeline.generator.generate_answer(context)
                predicted_choice = extract_answer_choice(result.answer)
                correct_choice = sample["answer_choice"]

                is_correct = (predicted_choice == correct_choice)
                if is_correct:
                    correct += 1

                logger.info(
                    f"  [{i+1}/{total}] Q: {sample['question'][:50]}... "
                    f"Predicted: {predicted_choice}, Correct: {correct_choice} "
                    f"{'✓' if is_correct else '✗'}"
                )

            except Exception as e:
                logger.error(f"  [{i+1}/{total}] Error: {e}")

        accuracy = correct / total if total > 0 else 0
        results[scenario_name] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }

        logger.info(f"{scenario_name} accuracy: {accuracy*100:.1f}% ({correct}/{total})")

    # Overall results
    total_correct = sum(r['correct'] for r in results.values())
    total_samples = sum(r['total'] for r in results.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {overall_accuracy*100:.1f}% ({total_correct}/{total_samples})")
    print(f"Target Range: 53-59%")
    print("\nPer-scenario results:")
    for scenario_name, result in results.items():
        print(f"  {scenario_name:12s}: {result['accuracy']*100:5.1f}% ({result['correct']}/{result['total']})")

    return results


if __name__ == "__main__":
    evaluate_mrag_bench(max_samples_per_scenario=10, use_gt_images=True)
