"""Quick test of multiple-choice evaluation (2 samples only)."""

import logging
from datasets import load_dataset
from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.generation.interface import MultimodalContext
import re

logging.basicConfig(level=logging.INFO)

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

# Load just 2 samples
print("Loading 2 test samples...")
dataset = load_dataset("uclanlp/MRAG-Bench", split="test[:2]")

for idx, sample in enumerate(dataset):
    print(f"\n{'='*80}")
    print(f"Sample {idx+1}: {sample['question']}")
    print(f"Scenario: {sample['scenario']}")

    choices = {
        "A": sample["A"],
        "B": sample["B"],
        "C": sample["C"],
        "D": sample["D"]
    }

    print("Choices:")
    for k, v in sorted(choices.items()):
        marker = "✓" if k == sample["answer_choice"] else " "
        print(f"  {marker} {k}. {v}")

    # Use GT images (first 3)
    images = sample["gt_images"][:3]
    print(f"Using {len(images)} GT images")

    context = MultimodalContext(
        question=sample["question"],
        images=images,
        choices=choices
    )

    # Generate
    print("\nGenerating answer...")
    result = pipeline.generator.generate_answer(context)
    predicted = extract_answer_choice(result.answer)

    print(f"Model response: '{result.answer}'")
    print(f"Extracted choice: {predicted}")
    print(f"Correct answer: {sample['answer_choice']}")
    print(f"Result: {'✓ CORRECT' if predicted == sample['answer_choice'] else '✗ WRONG'}")
