"""Test image-based retrieval to verify the fix works."""

from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.dataset import MRAGDataset
from pathlib import Path

# Load config
config = MRAGConfig.load('config/mrag_bench.yaml')
dataset = MRAGDataset(config.dataset.data_path)

# Get a test sample
samples = dataset.get_samples_by_scenario('scope')
sample = samples[0]  # First scope sample

print("="*80)
print("IMAGE-BASED RETRIEVAL TEST")
print("="*80)
print(f"\nTest Sample:")
print(f"  Question ID: {sample.question_id}")
print(f"  Question: {sample.question}")
print(f"  Query Image: {sample.image_path}")
print(f"  Ground Truth: {sample.ground_truth}")

# Initialize pipeline
pipeline = MRAGPipeline(config)
pipeline.load_retriever()

# Test 1: Image-based retrieval (correct approach)
print(f"\n{'='*80}")
print("TEST 1: IMAGE-BASED RETRIEVAL (MRAG-Bench format)")
print("="*80)

from PIL import Image
query_image = Image.open(sample.image_path).convert('RGB')
retrieval_results = pipeline.retriever.retrieve_by_image(query_image, k=10)

print(f"\nRetrieved {len(retrieval_results)} images using QUERY IMAGE:")
print("-" * 80)

# Extract question number from question_id
question_num = sample.question_id.replace('mrag_', '')

for i, result in enumerate(retrieval_results):
    img_path = Path(result.image_path)
    score = result.similarity_score
    filename = img_path.name

    print(f"\n{i+1}. {filename}")
    print(f"   Similarity score: {score:.4f}")

    # Check if this is from the same question
    if question_num in filename:
        print(f"   ✅ This is from the SAME question (relevant)!")

# Check results
relevant_count = sum(1 for r in retrieval_results if question_num in Path(r.image_path).name)
print(f"\n{'='*80}")
print("IMAGE-BASED RETRIEVAL RESULTS")
print("="*80)
print(f"Relevant images retrieved: {relevant_count}/10")

if relevant_count > 0:
    print(f"✅ SUCCESS: Found {relevant_count} relevant images using image-based retrieval!")
    print(f"   This should dramatically improve accuracy.")
else:
    print(f"❌ PROBLEM: No relevant images found even with image-based retrieval")

# Test 2: Compare with old text-based approach
print(f"\n{'='*80}")
print("TEST 2: TEXT-BASED RETRIEVAL (old, incorrect approach)")
print("="*80)

text_results = pipeline.retriever.retrieve_similar(sample.question, k=10)

print(f"\nRetrieved {len(text_results)} images using QUESTION TEXT:")
print("-" * 80)

for i, result in enumerate(text_results[:5]):  # Just show top 5
    img_path = Path(result.image_path)
    score = result.similarity_score
    filename = img_path.name

    print(f"\n{i+1}. {filename}")
    print(f"   Similarity score: {score:.4f}")

    if question_num in filename:
        print(f"   ✅ From same question")

text_relevant_count = sum(1 for r in text_results if question_num in Path(r.image_path).name)
print(f"\n{'='*80}")
print("COMPARISON")
print("="*80)
print(f"Image-based retrieval: {relevant_count}/10 relevant")
print(f"Text-based retrieval:  {text_relevant_count}/10 relevant")
print(f"\nImprovement: {relevant_count - text_relevant_count} more relevant images")

if relevant_count > text_relevant_count:
    print("✅ Image-based retrieval is BETTER - this is the fix we needed!")
elif relevant_count == text_relevant_count:
    print("⚠️  Both methods perform the same - need further investigation")
else:
    print("❌ Text-based is better - something wrong with image retrieval")
