"""Simple evaluation - first 40 samples without scenario filtering."""

import logging
import re
from datasets import load_dataset
from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.generation.interface import MultimodalContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_answer_choice(response: str) -> str:
    """Extract A/B/C/D from response."""
    response = response.strip().upper()
    if response in ['A', 'B', 'C', 'D']:
        return response
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]
    return ""


# Load config and pipeline
config = MRAGConfig.load('config/mrag_bench.yaml')
pipeline = MRAGPipeline(config)
pipeline.load_generator()

# Load first 40 samples
logger.info("Loading first 40 samples from MRAG-Bench...")
dataset = load_dataset("uclanlp/MRAG-Bench", split="test[:40]")
logger.info(f"Loaded {len(dataset)} samples")

correct = 0
total = len(dataset)

for idx, sample in enumerate(dataset):
    choices = {
        "A": sample["A"],
        "B": sample["B"],
        "C": sample["C"],
        "D": sample["D"]
    }

    # Use GT images (first 3)
    images = sample["gt_images"][:3]

    context = MultimodalContext(
        question=sample["question"],
        images=images,
        choices=choices
    )

    # Generate
    try:
        result = pipeline.generator.generate_answer(context)
        predicted = extract_answer_choice(result.answer)
        correct_answer = sample["answer_choice"]

        is_correct = (predicted == correct_answer)
        if is_correct:
            correct += 1

        logger.info(
            f"[{idx+1}/{total}] Scenario: {sample['scenario']:12s} | "
            f"Predicted: {predicted:1s} | Correct: {correct_answer:1s} | "
            f"{'✓' if is_correct else '✗'}"
        )

    except Exception as e:
        logger.error(f"[{idx+1}/{total}] Error: {e}")

accuracy = correct / total if total > 0 else 0

print("\n" + "="*80)
print("EVALUATION RESULTS (First 40 samples)")
print("="*80)
print(f"Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
print(f"Target Range: 53-59%")
print(f"Status: {'✓ PASS' if accuracy >= 0.53 else '✗ FAIL'}")
