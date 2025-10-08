"""Test multiple-choice format with HuggingFace dataset."""

from datasets import load_dataset
from src.generation.interface import MultimodalContext

print("Loading MRAG-Bench (just first 10 samples for testing)...")
dataset = load_dataset("uclanlp/MRAG-Bench", split="test[:10]")
print(f"Loaded {len(dataset)} samples\n")

# Test first sample
sample = dataset[0]

print("="*80)
print("SAMPLE FORMAT")
print("="*80)
print(f"Question: {sample['question']}")
print(f"\nChoices:")
print(f"  A. {sample['A']}")
print(f"  B. {sample['B']}")
print(f"  C. {sample['C']}")
print(f"  D. {sample['D']}")
print(f"\nCorrect Answer: {sample['answer_choice']} ({sample['answer']})")
print(f"Scenario: {sample['scenario']}")

# Create context
choices = {
    "A": sample["A"],
    "B": sample["B"],
    "C": sample["C"],
    "D": sample["D"]
}

context = MultimodalContext(
    question=sample["question"],
    images=sample["gt_images"][:3],  # Use first 3 GT images
    choices=choices
)

# Show what the prompt would look like
print("\n" + "="*80)
print("MULTIPLE-CHOICE PROMPT FORMAT")
print("="*80)

image_tokens = "<image>\n" * len(context.images)
choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])
prompt = (
    f"{image_tokens}"
    f"Answer the following multiple-choice question by selecting the correct option (A, B, C, or D).\n\n"
    f"Question: {context.question}\n\n"
    f"{choices_text}\n\n"
    f"Answer with only the letter (A, B, C, or D):\n"
)

print(prompt)
print("="*80)
print(f"\nExpected output: {sample['answer_choice']}")
print(f"We will match this against the model's generated output")
