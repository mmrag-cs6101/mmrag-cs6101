"""
Compare Object Detection Impact

This script runs evaluations with and without object detection to measure
the impact on visual reasoning accuracy.
"""

import os
import logging
from eval_enhanced import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_detection_impact(num_samples: int = 40):
    """
    Run comparison between standard and enhanced pipelines.
    
    Args:
        num_samples: Number of samples to evaluate
    """
    print("\n" + "="*80)
    print("OBJECT DETECTION IMPACT COMPARISON")
    print("="*80)
    print(f"Evaluating {num_samples} samples with and without object detection")
    print("="*80 + "\n")
    
    # Run without object detection (baseline)
    logger.info("Running BASELINE evaluation (without object detection)...")
    baseline_accuracy = run_evaluation(
        num_samples=num_samples,
        use_object_detection=False,
        save_results=True
    )
    
    print("\n" + "="*80)
    print("Baseline evaluation complete. Starting enhanced evaluation...")
    print("="*80 + "\n")
    
    # Run with object detection (enhanced)
    logger.info("Running ENHANCED evaluation (with object detection)...")
    enhanced_accuracy = run_evaluation(
        num_samples=num_samples,
        use_object_detection=True,
        save_results=True
    )
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"Baseline Accuracy (no detection):  {baseline_accuracy*100:.1f}%")
    print(f"Enhanced Accuracy (with detection): {enhanced_accuracy*100:.1f}%")
    print(f"Improvement: {(enhanced_accuracy - baseline_accuracy)*100:+.1f} percentage points")
    print(f"Relative Improvement: {((enhanced_accuracy / baseline_accuracy - 1) * 100):+.1f}%")
    print("="*80)
    
    # Determine if target was reached
    target_min = 0.53
    target_max = 0.59
    
    print("\nTarget Achievement:")
    print(f"  Target Range: {target_min*100:.0f}%-{target_max*100:.0f}%")
    print(f"  Baseline: {'✓ PASS' if baseline_accuracy >= target_min else '✗ FAIL'}")
    print(f"  Enhanced: {'✓ PASS' if enhanced_accuracy >= target_min else '✗ FAIL'}")
    
    if enhanced_accuracy >= target_min and baseline_accuracy < target_min:
        print("\n🎉 Object detection enabled reaching the target accuracy!")
    elif enhanced_accuracy > baseline_accuracy:
        print("\n✓ Object detection improved accuracy")
    else:
        print("\n⚠ Object detection did not improve accuracy")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare object detection impact on MRAG-Bench")
    parser.add_argument("--samples", type=int, default=40, help="Number of samples to evaluate (default: 40)")
    parser.add_argument("--full", action="store_true", help="Evaluate full dataset (1353 samples)")
    
    args = parser.parse_args()
    
    num_samples = None if args.full else args.samples
    
    if args.full:
        print("\n⚠️  WARNING: Full evaluation will take 4-6 hours!")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Evaluation cancelled.")
            exit(0)
    
    compare_detection_impact(num_samples=num_samples if num_samples else 1353)
