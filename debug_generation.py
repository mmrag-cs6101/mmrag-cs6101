"""Debug what LLaVA is generating with image-based retrieval."""

from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.dataset import MRAGDataset
from PIL import Image

# Load config
config = MRAGConfig.load('config/mrag_bench.yaml')
dataset = MRAGDataset(config.dataset.data_path)

# Get test samples
samples = dataset.get_samples_by_scenario('scope')[:3]  # Just 3 samples

# Initialize pipeline
pipeline = MRAGPipeline(config)
pipeline.initialize_dataset()
pipeline.load_retriever()
pipeline.load_generator()

print("="*80)
print("DEBUGGING GENERATION WITH IMAGE-BASED RETRIEVAL")
print("="*80)

for sample in samples:
    print(f"\n{'='*80}")
    print(f"Question ID: {sample.question_id}")
    print(f"Question: {sample.question}")
    print(f"Query Image: {sample.image_path}")
    print(f"Ground Truth: {sample.ground_truth}")
    print("-"*80)

    # Get retrieval results
    query_image = Image.open(sample.image_path).convert('RGB')
    retrieval_results = pipeline.retriever.retrieve_by_image(query_image, k=3)

    print("\nRetrieved Images:")
    for i, result in enumerate(retrieval_results):
        print(f"  {i+1}. {result.image_path} (score: {result.similarity_score:.4f})")

    # Generate answer
    result = pipeline.process_query(
        question=sample.question,
        question_id=sample.question_id,
        ground_truth=sample.ground_truth,
        query_image_path=sample.image_path
    )

    print(f"\nGenerated Answer: '{result.generated_answer}'")
    print(f"Ground Truth:     '{sample.ground_truth}'")
    print(f"Confidence: {result.confidence_score:.2f}")

    # Simple match check
    gen_lower = result.generated_answer.lower().replace('.', '').replace(',', '').strip()
    gt_lower = sample.ground_truth.lower().strip()

    is_correct = (gen_lower == gt_lower) or (gen_lower in gt_lower) or (gt_lower in gen_lower)
    print(f"Match: {'✅' if is_correct else '❌'}")

    if not is_correct:
        print("\n⚠️  MISMATCH - Why is this wrong?")
        print(f"   Generated: '{result.generated_answer}'")
        print(f"   Expected:  '{sample.ground_truth}'")
