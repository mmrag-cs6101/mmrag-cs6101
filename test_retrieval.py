"""Test retrieval quality - see what images are actually retrieved."""

from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.dataset import MRAGDataset
from pathlib import Path

# Load config
config = MRAGConfig.load('config/mrag_bench.yaml')
dataset = MRAGDataset(config.dataset.data_path)

# Get a test sample
samples = dataset.get_samples_by_scenario('angle')
sample = samples[0]  # First angle sample

print("="*80)
print("RETRIEVAL QUALITY TEST")
print("="*80)
print(f"\nTest Sample:")
print(f"  Question ID: {sample.question_id}")
print(f"  Question: {sample.question}")
print(f"  Ground Truth: {sample.ground_truth}")

# Initialize pipeline
pipeline = MRAGPipeline(config)
pipeline.load_retriever()

# Test retrieval
print(f"\nRetrieving top 10 images for query...")
retrieval_results = pipeline.retriever.retrieve_similar(
    sample.question,
    k=10
)

print(f"\nRetrieved {len(retrieval_results)} images:")
print("-" * 80)

for i, result in enumerate(retrieval_results):
    img_path = Path(result.image_path)
    score = result.similarity_score

    # Extract image info from filename
    filename = img_path.name

    print(f"\n{i+1}. {filename}")
    print(f"   Similarity score: {score:.4f}")
    print(f"   Full path: {result.image_path}")

    # Check if this is a ground truth image for this question
    if sample.question_id.replace('mrag_', '') in filename:
        print(f"   ✅ This is from the same question!")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Check if any retrieved images are from the same question
question_num = sample.question_id.replace('mrag_', '')
relevant_count = sum(1 for r in retrieval_results if question_num in Path(r.image_path).name)

print(f"\nRetrieved images from this question: {relevant_count}/10")

if relevant_count == 0:
    print("❌ PROBLEM: No relevant images retrieved!")
    print("   The retrieval is finding completely unrelated images.")
else:
    print(f"✅ Found {relevant_count} relevant images")

# Show score distribution
scores = [r.similarity_score for r in retrieval_results]
print(f"\nSimilarity score distribution:")
print(f"  Top score: {scores[0]:.4f}")
print(f"  Bottom score: {scores[-1]:.4f}")
print(f"  Score range: {scores[0] - scores[-1]:.4f}")

if scores[0] - scores[-1] < 0.01:
    print("  ⚠️  Warning: All scores very similar - retrieval may not be discriminative")
