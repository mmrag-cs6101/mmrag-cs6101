"""
Analyze actual generated answers vs ground truth to understand accuracy issues.
"""

import json
from pathlib import Path
import sys

def analyze_sprint10_results():
    """Analyze Sprint 10 validation results to understand failure patterns."""

    results_file = Path("output/sprint10/sprint10_final_validation_results.json")

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("SPRINT 10 VALIDATION RESULTS ANALYSIS")
    print("=" * 80)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total Queries: {data['total_questions']}")
    print(f"  Correct Answers: {data['total_correct']}")
    print(f"  Overall Accuracy: {data['overall_accuracy']:.1%}")
    print(f"  Target Range: {data['target_range'][0]:.1%} - {data['target_range'][1]:.1%}")

    print(f"\n📈 Performance Metrics:")
    perf = data['performance_metrics']
    print(f"  Total Queries Attempted: {perf['total_queries']}")
    print(f"  Successful (Correct): {perf['successful_queries']}")
    print(f"  Failed (Incorrect): {perf['failed_queries']}")
    print(f"  Success Rate: {perf['success_rate']:.1%}")
    print(f"  Error Rate: {perf['error_rate']:.1%}")

    print("\n⚠️  IMPORTANT CLARIFICATION:")
    print("  'successful_queries' = CORRECT answers")
    print("  'failed_queries' = INCORRECT answers (NOT system failures)")
    print("  'error_rate' = WRONG ANSWER rate (NOT crash rate)")

    print(f"\n🎯 Scenario Breakdown:")
    for scenario_name, scenario_data in data['scenario_results'].items():
        print(f"\n  {scenario_name.upper()}:")
        print(f"    Samples: {scenario_data['total_samples']}")
        print(f"    Correct: {scenario_data['correct_answers']}")
        print(f"    Accuracy: {scenario_data['accuracy']:.1%}")
        print(f"    Avg Time: {scenario_data['avg_processing_time']:.2f}s")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("\n✅ The system IS working - it's not crashing or failing")
    print("❌ The problem is LOW ACCURACY - generated answers don't match ground truth")
    print("\nThe 98.5% 'error rate' means:")
    print("  - Queries complete successfully")
    print("  - Answers are generated")
    print("  - But answers are WRONG 98.5% of the time")

    print("\n🔍 To improve accuracy, we need to:")
    print("  1. Check what answers are actually being generated")
    print("  2. Compare to ground truth to see the gap")
    print("  3. Improve prompt engineering / retrieval quality")
    print("  4. Possibly relax matching criteria (current: Jaccard >= 0.6)")

def find_sample_outputs():
    """Try to find individual sample outputs to analyze."""

    print("\n" + "=" * 80)
    print("SEARCHING FOR DETAILED OUTPUT EXAMPLES")
    print("=" * 80)

    # Check for detailed results files
    output_dir = Path("output")
    json_files = list(output_dir.rglob("*.json"))

    print(f"\nFound {len(json_files)} JSON files in output/")

    # Look for files with detailed results
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check if it has sample-level data
            if isinstance(data, dict):
                if 'test_results' in data and isinstance(data['test_results'], list):
                    if len(data['test_results']) > 0:
                        print(f"\n✅ Found detailed results in: {json_file}")
                        print(f"   Samples: {len(data['test_results'])}")

                        # Show first few examples
                        print("\n   Sample outputs (first 5):")
                        for i, result in enumerate(data['test_results'][:5]):
                            print(f"\n   Example {i+1}:")
                            print(f"     Question: {result.get('question', 'N/A')[:60]}...")
                            print(f"     Ground Truth: '{result.get('ground_truth', 'N/A')}'")
                            print(f"     Generated: '{result.get('generated_answer', 'N/A')[:60]}'")
                            print(f"     Correct: {result.get('is_correct', 'N/A')}")

                        return json_file
        except:
            continue

    print("\n❌ No detailed sample outputs found")
    print("   Need to run evaluation with detailed logging enabled")
    return None

if __name__ == "__main__":
    analyze_sprint10_results()
    find_sample_outputs()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Run a quick test to see actual generated answers:")
    print("   python run_sprint10_final_validation.py --quick-test --max-samples 5")
    print("\n2. Check if answers are:")
    print("   a) Completely irrelevant (retrieval problem)")
    print("   b) Close but not matching (prompt/generation problem)")
    print("   c) Correct but not matching criteria (evaluation problem)")
    print("\n3. Based on findings, apply targeted fixes")
