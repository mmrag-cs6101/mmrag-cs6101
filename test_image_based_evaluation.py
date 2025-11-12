"""Quick test to validate image-based retrieval improves accuracy."""

from src.config import MRAGConfig
from src.pipeline import MRAGPipeline
from src.dataset import MRAGDataset
from src.evaluation.evaluator import MRAGBenchEvaluator, PerspectiveChangeType
import logging

logging.basicConfig(level=logging.INFO)

# Load config
config = MRAGConfig.load('config/mrag_bench.yaml')

# Test on just 5 samples per scenario (20 total) for quick validation
print("="*80)
print("IMAGE-BASED RETRIEVAL VALIDATION TEST")
print("Quick test on 5 samples per scenario (20 total)")
print("="*80)

# Initialize evaluator
evaluator = MRAGBenchEvaluator(config)
evaluator.initialize_components()

# Test on small subset
test_results = {}
scenarios = [
    PerspectiveChangeType.ANGLE,
    PerspectiveChangeType.PARTIAL,
    PerspectiveChangeType.SCOPE,
    PerspectiveChangeType.OCCLUSION
]
for scenario in scenarios:
    print(f"\nTesting {scenario.value} scenario...")
    result = evaluator.evaluate_scenario(scenario, max_samples=5)
    test_results[scenario.value] = result
    print(f"  Accuracy: {result.accuracy*100:.1f}%")
    print(f"  Correct: {result.correct_answers}/{result.total_questions}")

# Overall results
total_samples = sum(r.total_questions for r in test_results.values())
total_correct = sum(r.correct_answers for r in test_results.values())
overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

print("\n" + "="*80)
print("VALIDATION RESULTS")
print("="*80)
print(f"Overall Accuracy: {overall_accuracy*100:.1f}% ({total_correct}/{total_samples})")
print(f"Target Range: 53-59%")

if overall_accuracy >= 0.30:
    print("\n✅ SIGNIFICANT IMPROVEMENT! Image-based retrieval is working!")
    print("   Ready to run full evaluation.")
elif overall_accuracy >= 0.10:
    print("\n⚠️  MODERATE IMPROVEMENT - better than before (2.5%) but below target")
    print("   May need additional prompt/parameter tuning.")
else:
    print("\n❌ Still very low accuracy - need to investigate further")

print("\nPer-scenario breakdown:")
for scenario, result in test_results.items():
    print(f"  {scenario}: {result.accuracy*100:.1f}% ({result.correct_answers}/{result.total_questions})")
