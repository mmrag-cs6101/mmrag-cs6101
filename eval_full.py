"""Full MRAG-Bench evaluation - all 1353 samples with detailed reporting."""

import os
import logging
import re
import json
import time
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.generation.interface import MultimodalContext

# Enable faster model downloads with hf_transfer (2-5x faster)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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

# Load full dataset (all 1353 samples)
logger.info("Loading full MRAG-Bench dataset from HuggingFace...")
dataset = load_dataset("uclanlp/MRAG-Bench", split="test")
logger.info(f"Loaded {len(dataset)} samples")

# Scenario mapping
scenario_mapping = {
    'Scope': 'scope',
    'Obstruction': 'scope',
    'Temporal': 'partial',
    'Deformation': 'scope',
    'Biological': 'angle',
    'Angle': 'angle',
    'Partial': 'partial',
    'Incomplete': 'partial',
    'Others': 'scope'
}

# Track results
correct = 0
total = len(dataset)
scenario_results = {}
start_time = time.time()

# Process each sample
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

    # Track by scenario
    scenario = sample["scenario"]
    perspective_type = scenario_mapping.get(scenario, "scope")
    if perspective_type not in scenario_results:
        scenario_results[perspective_type] = {"correct": 0, "total": 0}

    # Generate
    try:
        result = pipeline.generator.generate_answer(context)
        predicted = extract_answer_choice(result.answer)
        correct_answer = sample["answer_choice"]

        is_correct = (predicted == correct_answer)
        if is_correct:
            correct += 1
            scenario_results[perspective_type]["correct"] += 1

        scenario_results[perspective_type]["total"] += 1

        # Log every 50 samples
        if (idx + 1) % 50 == 0:
            current_accuracy = correct / (idx + 1) * 100
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (total - idx - 1)
            logger.info(
                f"Progress: [{idx+1}/{total}] | "
                f"Current Accuracy: {current_accuracy:.1f}% | "
                f"ETA: {eta/60:.1f}m"
            )
        else:
            logger.info(
                f"[{idx+1}/{total}] Scenario: {sample['scenario']:12s} | "
                f"Predicted: {predicted:1s} | Correct: {correct_answer:1s} | "
                f"{'✓' if is_correct else '✗'}"
            )

    except Exception as e:
        logger.error(f"[{idx+1}/{total}] Error: {e}")
        scenario_results[perspective_type]["total"] += 1

# Calculate results
accuracy = correct / total if total > 0 else 0
elapsed_time = time.time() - start_time

# Calculate per-scenario accuracy
scenario_accuracies = {}
for scenario, stats in scenario_results.items():
    scenario_accuracies[scenario] = {
        "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
        "correct": stats["correct"],
        "total": stats["total"]
    }

# Print results
print("\n" + "="*80)
print("FULL MRAG-BENCH EVALUATION RESULTS")
print("="*80)
print(f"Overall Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
print(f"Target Range: 53-59%")
print(f"Status: {'✓ PASS' if accuracy >= 0.53 else '✗ FAIL'}")
print(f"Evaluation Time: {elapsed_time/60:.1f} minutes")
print(f"Average Time per Sample: {elapsed_time/total:.2f}s")

print("\nPer-Scenario Results:")
print("-"*80)
for scenario in ['angle', 'partial', 'scope', 'occlusion']:
    if scenario in scenario_accuracies:
        stats = scenario_accuracies[scenario]
        print(
            f"  {scenario:12s}: {stats['accuracy']*100:5.1f}% "
            f"({stats['correct']}/{stats['total']})"
        )

# Save results
output_dir = Path("output/full_evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

results_data = {
    "timestamp": datetime.now().isoformat(),
    "overall_accuracy": accuracy,
    "total_samples": total,
    "correct_answers": correct,
    "target_range": [0.53, 0.59],
    "target_achieved": accuracy >= 0.53,
    "evaluation_time_seconds": elapsed_time,
    "avg_time_per_sample": elapsed_time / total,
    "scenario_results": scenario_accuracies,
    "configuration": {
        "model": config.model.vlm_name,
        "retriever": config.model.retriever_name,
        "quantization": config.model.quantization,
        "num_gt_images_used": 3,
        "max_length": config.generation.max_length,
        "temperature": config.generation.temperature
    }
}

results_file = output_dir / f"full_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\nResults saved to: {results_file}")
print("="*80)
