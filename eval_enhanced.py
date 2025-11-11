"""
Enhanced Evaluation with Object Detection

This script evaluates the MRAG-Bench system with integrated object detection
to improve visual reasoning accuracy.
"""

import os
import logging
import re
import json
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from src.config import MRAGConfig
from src.generation.llava_enhanced_pipeline import EnhancedLLaVAPipeline
from src.generation.interface import MultimodalContext, GenerationConfig

# Enable faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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


def run_evaluation(
    num_samples: int = 40,
    use_object_detection: bool = True,
    save_results: bool = True
):
    """
    Run enhanced evaluation with object detection.
    
    Args:
        num_samples: Number of samples to evaluate (40 for quick test, None for full)
        use_object_detection: Enable object detection enhancement
        save_results: Save detailed results to file
    """
    logger.info("="*80)
    logger.info("ENHANCED MRAG-BENCH EVALUATION WITH OBJECT DETECTION")
    logger.info("="*80)
    
    # Load configuration
    config = MRAGConfig.load('config/mrag_bench.yaml')
    
    # Create generation config
    gen_config = GenerationConfig(
        model_name=config.model.vlm_name,
        device=config.model.device,
        quantization=config.model.quantization,
        max_length=config.generation.max_length,
        temperature=config.generation.temperature,
        do_sample=config.generation.do_sample,
        max_memory_gb=config.performance.memory_limit_gb
    )
    
    # Initialize enhanced pipeline
    logger.info(f"Initializing enhanced pipeline (object detection: {use_object_detection})...")
    pipeline = EnhancedLLaVAPipeline(
        config=gen_config,
        use_object_detection=use_object_detection,
        detection_confidence=0.7,
        max_objects_per_image=10
    )
    
    # Load models
    logger.info("Loading models...")
    pipeline.load_model()
    logger.info("Models loaded successfully")
    
    # Load dataset
    if num_samples:
        logger.info(f"Loading first {num_samples} samples from MRAG-Bench...")
        dataset = load_dataset("uclanlp/MRAG-Bench", split=f"test[:{num_samples}]")
    else:
        logger.info("Loading full MRAG-Bench dataset...")
        dataset = load_dataset("uclanlp/MRAG-Bench", split="test")
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Evaluation metrics
    correct = 0
    total = len(dataset)
    results = []
    scenario_stats = {}
    
    # Track detection statistics
    total_detection_time = 0
    total_objects_detected = 0
    
    # Evaluate each sample
    logger.info("\nStarting evaluation...")
    logger.info("-"*80)
    
    for idx, sample in enumerate(dataset):
        scenario = sample["scenario"]
        
        # Prepare choices
        choices = {
            "A": sample["A"],
            "B": sample["B"],
            "C": sample["C"],
            "D": sample["D"]
        }
        
        # Use ground-truth images (first 3)
        images = sample["gt_images"][:3]
        
        # Create context
        context = MultimodalContext(
            question=sample["question"],
            images=images,
            choices=choices
        )
        
        # Generate answer
        try:
            result = pipeline.generate_answer(context)
            predicted = extract_answer_choice(result.answer)
            correct_answer = sample["answer_choice"]
            
            is_correct = (predicted == correct_answer)
            if is_correct:
                correct += 1
            
            # Update scenario statistics
            if scenario not in scenario_stats:
                scenario_stats[scenario] = {"correct": 0, "total": 0}
            scenario_stats[scenario]["total"] += 1
            if is_correct:
                scenario_stats[scenario]["correct"] += 1
            
            # Track detection statistics
            if result.metadata and result.metadata.get("detection_enabled"):
                if "detection_time" in result.metadata:
                    total_detection_time += result.metadata["detection_time"]
                if "total_objects_detected" in result.metadata:
                    total_objects_detected += result.metadata["total_objects_detected"]
            
            # Log progress
            logger.info(
                f"[{idx+1}/{total}] Scenario: {scenario:12s} | "
                f"Predicted: {predicted:1s} | Correct: {correct_answer:1s} | "
                f"{'✓' if is_correct else '✗'} | "
                f"Accuracy: {(correct/(idx+1))*100:.1f}%"
            )
            
            # Store detailed result
            result_entry = {
                "sample_id": idx,
                "scenario": scenario,
                "question": sample["question"],
                "predicted": predicted,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "generation_time": result.generation_time,
                "confidence": result.confidence_score
            }
            
            # Add detection info if available
            if result.metadata:
                if result.metadata.get("detection_enabled"):
                    result_entry["detection_info"] = {
                        "objects_detected": result.metadata.get("total_objects_detected", 0),
                        "detection_time": result.metadata.get("detection_time", 0),
                        "primary_objects": result.metadata.get("primary_objects", [])
                    }
            
            results.append(result_entry)
            
        except Exception as e:
            logger.error(f"[{idx+1}/{total}] Error: {e}")
            results.append({
                "sample_id": idx,
                "scenario": scenario,
                "error": str(e)
            })
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("ENHANCED EVALUATION RESULTS")
    print("="*80)
    print(f"Object Detection: {'ENABLED' if use_object_detection else 'DISABLED'}")
    print(f"Samples Evaluated: {total}")
    print(f"Overall Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    print(f"Target Range: 53-59%")
    print(f"Status: {'✓ PASS' if accuracy >= 0.53 else '✗ FAIL'}")
    
    if use_object_detection:
        avg_detection_time = total_detection_time / total if total > 0 else 0
        avg_objects = total_objects_detected / total if total > 0 else 0
        print(f"\nObject Detection Statistics:")
        print(f"  Total Objects Detected: {total_objects_detected}")
        print(f"  Avg Objects per Sample: {avg_objects:.1f}")
        print(f"  Avg Detection Time: {avg_detection_time:.2f}s")
    
    print(f"\nPer-Scenario Results:")
    for scenario, stats in sorted(scenario_stats.items()):
        scenario_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {scenario:12s}: {scenario_acc*100:5.1f}% ({stats['correct']}/{stats['total']})")
    
    # Save results if requested
    if save_results:
        output_dir = Path("output/enhanced_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detection_suffix = "with_detection" if use_object_detection else "without_detection"
        output_file = output_dir / f"eval_{detection_suffix}_{timestamp}.json"
        
        output_data = {
            "timestamp": timestamp,
            "object_detection_enabled": use_object_detection,
            "overall_accuracy": accuracy,
            "total_samples": total,
            "correct_answers": correct,
            "target_range": [0.53, 0.59],
            "target_achieved": accuracy >= 0.53,
            "scenario_results": {
                scenario: {
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                    "correct": stats["correct"],
                    "total": stats["total"]
                }
                for scenario, stats in scenario_stats.items()
            },
            "detailed_results": results
        }
        
        if use_object_detection:
            output_data["detection_statistics"] = {
                "total_objects_detected": total_objects_detected,
                "avg_objects_per_sample": total_objects_detected / total if total > 0 else 0,
                "total_detection_time": total_detection_time,
                "avg_detection_time": total_detection_time / total if total > 0 else 0
            }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    print("="*80)
    
    # Cleanup
    pipeline.unload_model()
    
    return accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced MRAG-Bench evaluation with object detection")
    parser.add_argument("--samples", type=int, default=40, help="Number of samples to evaluate (default: 40)")
    parser.add_argument("--full", action="store_true", help="Evaluate full dataset (1353 samples)")
    parser.add_argument("--no-detection", action="store_true", help="Disable object detection")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    num_samples = None if args.full else args.samples
    use_detection = not args.no_detection
    save_results = not args.no_save
    
    run_evaluation(
        num_samples=num_samples,
        use_object_detection=use_detection,
        save_results=save_results
    )
