"""
Dataset Validation and Integrity Checks

Comprehensive validation system for MRAG-Bench dataset integrity and quality.
"""

import logging
import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
from dataclasses import dataclass, asdict
import time

from .mrag_dataset import MRAGDataset
from .interface import Sample
try:
    from ..utils.memory_manager import MemoryManager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    status: str  # 'success', 'warning', 'error'
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    total_images: int = 0
    valid_images: int = 0
    corrupted_images: int = 0
    missing_images: int = 0
    scenario_distribution: Dict[str, int] = None
    image_statistics: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    validation_time: float = 0.0

    def __post_init__(self):
        if self.scenario_distribution is None:
            self.scenario_distribution = {}
        if self.image_statistics is None:
            self.image_statistics = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class DatasetValidator:
    """
    Comprehensive dataset validation and integrity checking.

    Validates:
    - Dataset structure and organization
    - Image file integrity and accessibility
    - Question-answer pair completeness
    - Perspective change scenario distribution
    - Image quality and consistency
    - Memory efficiency during processing
    """

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize dataset validator.

        Args:
            memory_manager: Optional memory manager for validation operations
        """
        self.memory_manager = memory_manager or MemoryManager()
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def validate_dataset(
        self,
        dataset: MRAGDataset,
        check_images: bool = True,
        check_image_quality: bool = True,
        sample_limit: Optional[int] = None
    ) -> ValidationResult:
        """
        Perform comprehensive dataset validation.

        Args:
            dataset: MRAG dataset to validate
            check_images: Whether to validate image files
            check_image_quality: Whether to check image quality metrics
            sample_limit: Optional limit on number of samples to validate

        Returns:
            Validation result with detailed statistics
        """
        start_time = time.time()
        result = ValidationResult()

        logger.info("Starting comprehensive dataset validation...")

        try:
            with self.memory_manager.memory_guard("dataset_validation"):
                # Validate dataset structure
                self._validate_structure(dataset, result)

                # Validate questions and metadata
                self._validate_questions(dataset, result, sample_limit)

                # Validate images if requested
                if check_images:
                    self._validate_images(dataset, result, check_image_quality, sample_limit)

                # Validate scenario distribution
                self._validate_scenarios(dataset, result)

                # Determine overall status
                self._determine_status(result)

        except Exception as e:
            result.status = "error"
            result.errors.append(f"Validation failed: {str(e)}")
            logger.error(f"Dataset validation failed: {e}")

        result.validation_time = time.time() - start_time
        logger.info(f"Dataset validation completed in {result.validation_time:.2f} seconds")

        return result

    def _validate_structure(self, dataset: MRAGDataset, result: ValidationResult) -> None:
        """Validate dataset directory structure."""
        logger.info("Validating dataset structure...")

        required_dirs = ['images', 'questions', 'metadata']
        missing_dirs = []

        for dir_name in required_dirs:
            dir_path = dataset.data_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            result.warnings.append(f"Missing directories: {missing_dirs}")

        # Check for required files
        required_files = [
            'questions/questions.json',
            'metadata/dataset_info.json'
        ]

        for file_path in required_files:
            full_path = dataset.data_path / file_path
            if not full_path.exists():
                result.warnings.append(f"Missing file: {file_path}")

    def _validate_questions(
        self,
        dataset: MRAGDataset,
        result: ValidationResult,
        sample_limit: Optional[int]
    ) -> None:
        """Validate question data completeness and format."""
        logger.info("Validating questions and metadata...")

        result.total_samples = len(dataset.questions_data)
        samples_to_check = dataset.questions_data[:sample_limit] if sample_limit else dataset.questions_data

        valid_count = 0
        invalid_count = 0

        for i, question_data in enumerate(samples_to_check):
            is_valid = True
            errors = []

            # Check required fields
            required_fields = ['question_id', 'question', 'answer']
            for field in required_fields:
                if field not in question_data or not question_data[field]:
                    errors.append(f"Missing or empty {field}")
                    is_valid = False

            # Check question quality
            question = question_data.get('question', '')
            if len(question.strip()) < 10:
                errors.append("Question too short")
                is_valid = False

            # Check answer format
            answer = question_data.get('answer', '')
            if not answer or len(answer.strip()) == 0:
                errors.append("Missing answer")
                is_valid = False

            # Check choices if present
            choices = question_data.get('choices', [])
            if choices and len(choices) < 2:
                errors.append("Insufficient answer choices")
                is_valid = False

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if len(result.errors) < 10:  # Limit error reporting
                    result.errors.extend([f"Sample {i}: {e}" for e in errors])

        result.valid_samples = valid_count
        result.invalid_samples = invalid_count

        if invalid_count > 0:
            result.warnings.append(f"Found {invalid_count} invalid samples out of {len(samples_to_check)}")

    def _validate_images(
        self,
        dataset: MRAGDataset,
        result: ValidationResult,
        check_quality: bool,
        sample_limit: Optional[int]
    ) -> None:
        """Validate image files and quality."""
        logger.info("Validating images...")

        # Get all image paths
        image_paths = dataset.get_retrieval_corpus()
        result.total_images = len(image_paths)

        if sample_limit:
            image_paths = image_paths[:sample_limit]

        valid_images = 0
        corrupted_images = 0
        missing_images = 0

        image_stats = {
            'sizes': [],
            'formats': {},
            'channels': {},
            'mean_brightness': [],
            'file_sizes': []
        }

        for i, image_path in enumerate(image_paths):
            try:
                if not os.path.exists(image_path):
                    missing_images += 1
                    continue

                # Check file size
                file_size = os.path.getsize(image_path)
                image_stats['file_sizes'].append(file_size)

                if file_size == 0:
                    corrupted_images += 1
                    result.errors.append(f"Empty image file: {image_path}")
                    continue

                # Try to open and validate image
                with Image.open(image_path) as img:
                    # Basic validation
                    if img.size[0] == 0 or img.size[1] == 0:
                        corrupted_images += 1
                        result.errors.append(f"Invalid image dimensions: {image_path}")
                        continue

                    # Collect statistics
                    image_stats['sizes'].append(img.size)

                    format_name = img.format or 'unknown'
                    image_stats['formats'][format_name] = image_stats['formats'].get(format_name, 0) + 1

                    mode = img.mode
                    image_stats['channels'][mode] = image_stats['channels'].get(mode, 0) + 1

                    # Quality checks if requested
                    if check_quality:
                        # Convert to RGB for analysis
                        rgb_img = img.convert('RGB')
                        img_array = np.array(rgb_img)

                        # Calculate brightness
                        brightness = np.mean(img_array)
                        image_stats['mean_brightness'].append(brightness)

                        # Check for extremely dark or bright images
                        if brightness < 30:
                            result.warnings.append(f"Very dark image: {image_path}")
                        elif brightness > 220:
                            result.warnings.append(f"Very bright image: {image_path}")

                    valid_images += 1

            except Exception as e:
                corrupted_images += 1
                if len(result.errors) < 20:  # Limit error reporting
                    result.errors.append(f"Corrupted image {image_path}: {str(e)}")

            # Periodic memory cleanup
            if i % 100 == 0:
                self.memory_manager.clear_cpu_memory()

        result.valid_images = valid_images
        result.corrupted_images = corrupted_images
        result.missing_images = missing_images

        # Calculate image statistics
        if image_stats['sizes']:
            widths, heights = zip(*image_stats['sizes'])
            result.image_statistics = {
                'total_analyzed': len(image_stats['sizes']),
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'formats': image_stats['formats'],
                'channels': image_stats['channels'],
                'avg_file_size_mb': np.mean(image_stats['file_sizes']) / (1024 * 1024),
                'total_size_gb': sum(image_stats['file_sizes']) / (1024**3)
            }

            if image_stats['mean_brightness']:
                result.image_statistics['avg_brightness'] = np.mean(image_stats['mean_brightness'])
                result.image_statistics['brightness_std'] = np.std(image_stats['mean_brightness'])

    def _validate_scenarios(self, dataset: MRAGDataset, result: ValidationResult) -> None:
        """Validate perspective change scenario distribution."""
        logger.info("Validating scenario distribution...")

        scenario_stats = dataset.get_scenario_stats()
        result.scenario_distribution = scenario_stats

        total_scenario_samples = sum(scenario_stats.values())
        min_samples_per_scenario = 50  # Minimum samples needed per scenario

        for scenario, count in scenario_stats.items():
            if count < min_samples_per_scenario:
                result.warnings.append(
                    f"Low sample count for scenario '{scenario}': {count} < {min_samples_per_scenario}"
                )

        # Check distribution balance
        if total_scenario_samples > 0:
            max_count = max(scenario_stats.values())
            min_count = min(scenario_stats.values())

            if max_count > 0 and min_count / max_count < 0.1:  # Very imbalanced
                result.warnings.append(
                    f"Highly imbalanced scenario distribution: {scenario_stats}"
                )

    def _determine_status(self, result: ValidationResult) -> None:
        """Determine overall validation status."""
        if result.errors:
            result.status = "error"
        elif result.warnings:
            result.status = "warning"
        else:
            result.status = "success"

    def validate_sample_batch(
        self,
        dataset: MRAGDataset,
        scenario_type: str,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Validate a small batch of samples for quick testing.

        Args:
            dataset: MRAG dataset
            scenario_type: Scenario to test
            batch_size: Number of samples to validate

        Returns:
            Quick validation results
        """
        logger.info(f"Validating sample batch for scenario: {scenario_type}")

        results = {
            "scenario": scenario_type,
            "samples_tested": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "preprocessing_successful": 0,
            "preprocessing_failed": 0,
            "errors": []
        }

        try:
            sample_count = 0
            for sample in dataset.load_scenario(scenario_type):
                if sample_count >= batch_size:
                    break

                results["samples_tested"] += 1

                # Test sample loading
                try:
                    if sample.image_path and os.path.exists(sample.image_path):
                        img = Image.open(sample.image_path)
                        img.verify()  # Verify image integrity
                        results["successful_loads"] += 1
                    else:
                        results["failed_loads"] += 1
                        results["errors"].append(f"Missing image: {sample.image_path}")

                except Exception as e:
                    results["failed_loads"] += 1
                    results["errors"].append(f"Image load error: {str(e)}")

                # Test preprocessing
                try:
                    batch_data = dataset.preprocess_batch([sample])
                    if batch_data and len(batch_data.images) > 0:
                        results["preprocessing_successful"] += 1
                    else:
                        results["preprocessing_failed"] += 1

                except Exception as e:
                    results["preprocessing_failed"] += 1
                    results["errors"].append(f"Preprocessing error: {str(e)}")

                sample_count += 1

        except Exception as e:
            results["errors"].append(f"Batch validation failed: {str(e)}")

        return results

    def generate_validation_report(self, result: ValidationResult, output_path: str) -> None:
        """
        Generate detailed validation report.

        Args:
            result: Validation result
            output_path: Path to save the report
        """
        report = {
            "validation_summary": {
                "status": result.status,
                "validation_time": result.validation_time,
                "total_samples": result.total_samples,
                "valid_samples": result.valid_samples,
                "total_images": result.total_images,
                "valid_images": result.valid_images
            },
            "detailed_results": result.to_dict(),
            "recommendations": self._generate_recommendations(result)
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to: {output_path}")

    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if result.invalid_samples > 0:
            recommendations.append(
                f"Clean up {result.invalid_samples} invalid samples to improve data quality"
            )

        if result.corrupted_images > 0:
            recommendations.append(
                f"Fix or remove {result.corrupted_images} corrupted images"
            )

        if result.missing_images > 0:
            recommendations.append(
                f"Locate and restore {result.missing_images} missing images"
            )

        # Check scenario balance
        if result.scenario_distribution:
            max_count = max(result.scenario_distribution.values())
            min_count = min(result.scenario_distribution.values())

            if max_count > 0 and min_count / max_count < 0.5:
                recommendations.append(
                    "Consider data augmentation for underrepresented scenarios"
                )

        if not recommendations:
            recommendations.append("Dataset validation passed successfully!")

        return recommendations