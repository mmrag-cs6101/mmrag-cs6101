#!/usr/bin/env python3
"""
Sprint 10: Final Accuracy Validation - Orchestration Script

Comprehensive final validation of MRAG-Bench system with:
- Complete evaluation on all 778 perspective change samples
- Multi-run validation for statistical confidence
- Comprehensive performance and reliability verification
- Production readiness assessment
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import MRAGConfig
from src.evaluation.final_validator import FinalAccuracyValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sprint10_final_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sprint 10: Final Accuracy Validation for MRAG-Bench System"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/mrag_bench.yaml',
        help='Path to configuration file (default: config/mrag_bench.yaml)'
    )

    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of evaluation runs for statistical validation (default: 3)'
    )

    parser.add_argument(
        '--full-dataset',
        action='store_true',
        default=True,
        help='Use full dataset (all 778 samples) (default: True)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per scenario (overrides --full-dataset if specified)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/sprint10',
        help='Output directory for results (default: output/sprint10)'
    )

    parser.add_argument(
        '--target-min',
        type=float,
        default=0.53,
        help='Target minimum accuracy (default: 0.53 = 53%%)'
    )

    parser.add_argument(
        '--target-max',
        type=float,
        default=0.59,
        help='Target maximum accuracy (default: 0.59 = 59%%)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode with small sample sizes (for testing)'
    )

    return parser.parse_args()


def run_final_validation(
    config_path: str,
    num_runs: int,
    full_dataset: bool,
    max_samples: Optional[int],
    output_dir: str,
    target_range: tuple,
    quick_test: bool = False
):
    """
    Run comprehensive final validation.

    Args:
        config_path: Path to configuration file
        num_runs: Number of evaluation runs
        full_dataset: Use full dataset
        max_samples: Maximum samples per scenario
        output_dir: Output directory
        target_range: Target accuracy range (min, max)
        quick_test: Quick test mode
    """
    logger.info("\n" + "="*70)
    logger.info("SPRINT 10: FINAL ACCURACY VALIDATION")
    logger.info("="*70)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Evaluation Runs: {num_runs}")
    logger.info(f"Target Range: {target_range[0]:.1%} - {target_range[1]:.1%}")
    logger.info(f"Full Dataset: {full_dataset}")
    if max_samples:
        logger.info(f"Max Samples Per Scenario: {max_samples}")
    if quick_test:
        logger.info("⚠️  QUICK TEST MODE - Results are for testing only!")
    logger.info("="*70 + "\n")

    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}...")
        config = MRAGConfig.from_yaml(config_path)
        logger.info("Configuration loaded successfully")

        # Override settings for quick test
        if quick_test:
            logger.info("Quick test mode enabled - overriding settings...")
            num_runs = 1
            max_samples = 10
            full_dataset = False

        # Initialize validator
        logger.info("Initializing Final Accuracy Validator...")
        validator = FinalAccuracyValidator(
            config=config,
            target_range=target_range,
            output_dir=output_dir
        )
        logger.info("Validator initialized successfully\n")

        # Run comprehensive validation
        logger.info("Starting comprehensive final validation...")
        logger.info(f"This will evaluate:")
        if full_dataset and not max_samples:
            logger.info("  - All 778 perspective change samples (complete dataset)")
            logger.info("  - Angle changes: ~322 samples")
            logger.info("  - Partial views: ~246 samples")
            logger.info("  - Scope variations: ~102 samples")
            logger.info("  - Occlusion: ~108 samples")
        elif max_samples:
            logger.info(f"  - Up to {max_samples} samples per scenario")
        logger.info(f"  - {num_runs} evaluation run(s) for statistical confidence")
        logger.info("")

        results = validator.run_comprehensive_validation(
            num_runs=num_runs,
            full_dataset=full_dataset,
            max_samples_per_scenario=max_samples
        )

        # Log detailed results
        logger.info("\n" + "="*70)
        logger.info("FINAL VALIDATION RESULTS")
        logger.info("="*70)

        logger.info(f"\nOverall Performance:")
        logger.info(f"  Overall Accuracy: {results.overall_accuracy:.1%}")
        logger.info(
            f"  95% Confidence Interval: "
            f"[{results.overall_confidence_interval[0]:.1%}, "
            f"{results.overall_confidence_interval[1]:.1%}]"
        )
        logger.info(f"  Total Questions: {results.total_questions}")
        logger.info(f"  Total Correct: {results.total_correct}")

        logger.info(f"\nTarget Achievement:")
        logger.info(f"  Target Range: {results.target_range[0]:.1%} - {results.target_range[1]:.1%}")
        logger.info(f"  Target Achieved: {'✓ YES' if results.target_achieved else '✗ NO'}")
        logger.info(f"  Scenarios in Target: {results.scenarios_in_target}/4")

        logger.info(f"\nScenario Breakdown:")
        for scenario_name, scenario_result in results.scenario_results.items():
            status = "✓" if scenario_result.in_target_range else "✗"
            logger.info(
                f"  {scenario_name.upper():10}: {scenario_result.accuracy:.1%} "
                f"({scenario_result.correct_answers}/{scenario_result.total_samples}) "
                f"- {status}"
            )

        if results.multi_run_statistics:
            logger.info(f"\nMulti-Run Statistics:")
            mrs = results.multi_run_statistics
            logger.info(f"  Mean ± Std: {mrs.mean_accuracy:.1%} ± {mrs.std_accuracy:.1%}")
            logger.info(f"  Range: [{mrs.min_accuracy:.1%}, {mrs.max_accuracy:.1%}]")
            logger.info(f"  CV: {mrs.coefficient_of_variation:.1%}")
            logger.info(f"  Cross-Run Consistency: {results.cross_run_consistency:.1%}")

        logger.info(f"\nPerformance Metrics:")
        perf = results.performance_metrics
        logger.info(f"  Avg Query Time: {perf.avg_query_time:.2f}s")
        logger.info(f"  P95 Query Time: {perf.p95_query_time:.2f}s")
        logger.info(f"  Peak Memory: {perf.peak_memory_gb:.2f}GB")
        logger.info(f"  Success Rate: {perf.success_rate:.1%}")

        logger.info(f"\nSystem Assessment:")
        logger.info(f"  Statistical Confidence: {results.statistical_confidence.upper()}")
        logger.info(f"  Production Readiness: {results.production_readiness.upper()}")

        logger.info(f"\nRecommendations:")
        for i, rec in enumerate(results.recommendations, 1):
            logger.info(f"  {i}. {rec}")

        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"  - JSON: sprint10_final_validation_results.json")
        logger.info(f"  - Report: sprint10_summary_report.md")

        logger.info("\n" + "="*70)
        logger.info("SPRINT 10 FINAL VALIDATION COMPLETE")
        logger.info("="*70 + "\n")

        # Cleanup
        validator.cleanup()

        # Return success/failure
        return 0 if results.target_achieved else 1

    except Exception as e:
        logger.error(f"\n❌ Final validation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    args = parse_args()

    # Prepare target range
    target_range = (args.target_min, args.target_max)

    # Run validation
    exit_code = run_final_validation(
        config_path=args.config,
        num_runs=args.num_runs,
        full_dataset=args.full_dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        target_range=target_range,
        quick_test=args.quick_test
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
