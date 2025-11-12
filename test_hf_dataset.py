"""Test HuggingFace MRAG-Bench dataset integration."""

from src.dataset.mrag_hf_dataset import MRAGHFDataset, MRAGSample
from src.config import MRAGConfig
from src.generation.llava_pipeline import LLaVAGenerationPipeline
from src.generation.interface import MultimodalContext

# Load dataset
print("Loading MRAG-Bench from HuggingFace...")
dataset = MRAGHFDataset(split="test", use_retrieved=False)  # Use GT images
print(f"Loaded {len(dataset)} samples\n")

# Get a test sample
sample = dataset[0]
print("="*80)
print("SAMPLE INFO")
print("="*80)
print(f"Question ID: {sample.question_id}")
print(f"Question: {sample.question}")
print(f"Choices:")
for k, v in sorted(sample.choices.items()):
    marker = "✓" if k == sample.answer_choice else " "
    print(f"  {marker} {k}. {v}")
print(f"\nCorrect Answer: {sample.answer_choice} ({sample.answer})")
print(f"Scenario: {sample.scenario}")
print(f"Query Image: {sample.query_image.size if sample.query_image else 'None'}")
print(f"GT Images: {len(sample.gt_images)}")
print(f"Retrieved Images: {len(sample.retrieved_images)}")

# Test prompt construction
print("\n" + "="*80)
print("PROMPT CONSTRUCTION TEST")
print("="*80)

# Create context with GT images
context = MultimodalContext(
    question=sample.question,
    images=sample.gt_images[:3],  # Use top 3 GT images
    choices=sample.choices
)

# Manually construct prompt to show format
image_tokens = "<image>\n" * len(context.images)
choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(context.choices.items())])
prompt = (
    f"{image_tokens}"
    f"Answer the following multiple-choice question by selecting the correct option (A, B, C, or D).\n\n"
    f"Question: {context.question}\n\n"
    f"{choices_text}\n\n"
    f"Answer with only the letter (A, B, C, or D):\n"
)

print("\nGenerated Prompt:")
print("-"*80)
print(prompt)
print("-"*80)

# Test scenario filtering
print("\n" + "="*80)
print("SCENARIO FILTERING TEST")
print("="*80)

for scenario in ['angle', 'partial', 'scope', 'occlusion']:
    samples = dataset.get_samples_by_scenario(scenario, max_samples=5)
    print(f"{scenario}: {len(samples)} samples (showing 5)")
    if samples:
        print(f"  Example: {samples[0].question[:50]}...")
