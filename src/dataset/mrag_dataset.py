"""
MRAG-Bench Dataset Implementation

Concrete implementation of DatasetInterface for MRAG-Bench dataset processing.
"""

import os
import json
import logging
from typing import Iterator, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from .interface import DatasetInterface, Sample, BatchData


logger = logging.getLogger(__name__)


class MRAGDataset(DatasetInterface):
    """
    MRAG-Bench dataset implementation with perspective change scenario filtering.

    Provides memory-efficient access to MRAG-Bench data with support for:
    - Perspective change scenario filtering (angle, partial, scope, occlusion)
    - Streaming batch processing
    - Image preprocessing with CLIP normalization
    - Comprehensive dataset validation
    """

    def __init__(self, data_path: str, batch_size: int = 8, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize MRAG-Bench dataset.

        Args:
            data_path: Path to MRAG-Bench dataset directory
            batch_size: Batch size for streaming operations
            image_size: Target image size for preprocessing (width, height)
        """
        super().__init__(data_path, batch_size)
        self.image_size = image_size
        self.data_path = Path(data_path)

        # Define perspective change scenarios based on MRAG-Bench
        # These will be mapped from actual dataset categories
        self.perspective_scenarios = {
            'angle': 'Angle Change',
            'partial': 'Partial View',
            'scope': 'Scope Variation',
            'occlusion': 'Occlusion'
        }

        # Load dataset metadata
        self._load_metadata()

        # Initialize image preprocessing pipeline
        self._setup_image_transforms()

        logger.info(f"MRAGDataset initialized with {self.total_samples} samples")

    def _load_metadata(self) -> None:
        """Load dataset metadata and question data."""
        try:
            # Load questions data
            questions_file = self.data_path / "questions" / "questions.json"
            if questions_file.exists():
                with open(questions_file, 'r') as f:
                    self.questions_data = json.load(f)
            else:
                logger.warning(f"Questions file not found: {questions_file}")
                self.questions_data = []

            # Load dataset metadata
            metadata_file = self.data_path / "metadata" / "dataset_info.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                logger.warning(f"Metadata file not found: {metadata_file}")
                self.metadata = {"scenarios": {}, "total_samples": len(self.questions_data)}

            # Load scenario mapping
            scenario_file = self.data_path / "metadata" / "scenario_mapping.json"
            if scenario_file.exists():
                with open(scenario_file, 'r') as f:
                    self.scenario_mapping = json.load(f)
            else:
                logger.warning(f"Scenario mapping not found: {scenario_file}")
                self.scenario_mapping = {}

            self.total_samples = len(self.questions_data)

            # Map actual categories to perspective change scenarios
            self._map_perspective_scenarios()

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.questions_data = []
            self.metadata = {"scenarios": {}, "total_samples": 0}
            self.scenario_mapping = {}
            self.total_samples = 0

    def _map_perspective_scenarios(self) -> None:
        """Map dataset categories to perspective change scenarios."""
        self.category_to_perspective = {}

        # Analyze existing categories to map to perspective changes
        if self.scenario_mapping:
            categories = list(self.scenario_mapping.keys())
            logger.info(f"Found categories: {categories}")

            # Create mapping from categories to perspective types
            # This is based on common patterns in vision benchmarks
            for category in categories:
                category_lower = category.lower()
                if any(word in category_lower for word in ['angle', 'rotation', 'viewpoint']):
                    self.category_to_perspective[category] = 'angle'
                elif any(word in category_lower for word in ['partial', 'crop', 'truncate']):
                    self.category_to_perspective[category] = 'partial'
                elif any(word in category_lower for word in ['scope', 'zoom', 'scale']):
                    self.category_to_perspective[category] = 'scope'
                elif any(word in category_lower for word in ['occlusion', 'hidden', 'blocked']):
                    self.category_to_perspective[category] = 'occlusion'
                else:
                    # Default mapping - distribute evenly across perspective types
                    perspective_types = ['angle', 'partial', 'scope', 'occlusion']
                    idx = hash(category) % len(perspective_types)
                    self.category_to_perspective[category] = perspective_types[idx]

        logger.info(f"Category to perspective mapping: {self.category_to_perspective}")

    def _setup_image_transforms(self) -> None:
        """Setup image preprocessing transforms for CLIP compatibility."""
        # CLIP preprocessing pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Basic preprocessing without normalization (for display/debugging)
        self.basic_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size)
        ])

    def load_scenario(self, scenario_type: str) -> Iterator[Sample]:
        """
        Load samples for specific perspective change scenario.

        Args:
            scenario_type: One of ['angle', 'partial', 'scope', 'occlusion']

        Yields:
            Sample objects for the specified scenario
        """
        if scenario_type not in self.perspective_types:
            raise ValueError(f"Invalid scenario type: {scenario_type}. Must be one of {self.perspective_types}")

        logger.info(f"Loading scenario: {scenario_type}")

        for question_data in self.questions_data:
            # Map category to perspective type
            category = question_data.get('scenario', question_data.get('category', 'unknown'))
            perspective_type = self.category_to_perspective.get(category, 'angle')  # Default to angle

            if perspective_type == scenario_type:
                try:
                    sample = self._create_sample_from_question(question_data)
                    if sample:
                        yield sample
                except Exception as e:
                    logger.warning(f"Error creating sample for question {question_data.get('question_id', 'unknown')}: {e}")
                    continue

    def _create_sample_from_question(self, question_data: Dict[str, Any]) -> Optional[Sample]:
        """Create a Sample object from question data."""
        try:
            question_id = question_data.get('question_id', '')
            question = question_data.get('question', '')
            image_path = question_data.get('image_path', '')
            ground_truth = question_data.get('answer', '')
            category = question_data.get('scenario', question_data.get('category', 'unknown'))

            # Map category to perspective type
            perspective_type = self.category_to_perspective.get(category, 'angle')

            # Convert relative image path to absolute
            if image_path:
                full_image_path = str(self.data_path / image_path)
            else:
                full_image_path = ""

            # Create metadata
            metadata = {
                'choices': question_data.get('choices', []),
                'category': category,
                'question_id': question_id
            }

            sample = Sample(
                question_id=question_id,
                question=question,
                image_path=full_image_path,
                image=None,  # Loaded on demand
                ground_truth=ground_truth,
                perspective_type=perspective_type,
                metadata=metadata
            )

            return sample

        except Exception as e:
            logger.error(f"Error creating sample: {e}")
            return None

    def get_retrieval_corpus(self) -> List[str]:
        """
        Get image paths for retrieval corpus.

        Returns:
            List of image file paths for building retrieval index
        """
        image_paths = []
        images_dir = self.data_path / "images"

        if images_dir.exists():
            # Get all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(str(p) for p in images_dir.glob(ext))

        logger.info(f"Found {len(image_paths)} images for retrieval corpus")
        return sorted(image_paths)

    def preprocess_batch(self, samples: List[Sample]) -> BatchData:
        """
        Preprocess a batch of samples for model input.

        Args:
            samples: List of Sample objects

        Returns:
            BatchData with preprocessed images and questions
        """
        processed_images = []
        questions = []

        for sample in samples:
            # Load and preprocess image
            if sample.image is None and sample.image_path and os.path.exists(sample.image_path):
                try:
                    image = Image.open(sample.image_path).convert('RGB')
                    sample.image = image
                except Exception as e:
                    logger.warning(f"Failed to load image {sample.image_path}: {e}")
                    # Create a placeholder image
                    sample.image = Image.new('RGB', self.image_size, color=(128, 128, 128))

            if sample.image:
                # Apply basic preprocessing (without normalization for now)
                processed_image = self.basic_transform(sample.image)
                processed_images.append(processed_image)
            else:
                # Create placeholder
                placeholder = Image.new('RGB', self.image_size, color=(128, 128, 128))
                processed_images.append(self.basic_transform(placeholder))

            questions.append(sample.question)

        return BatchData(
            samples=samples,
            images=processed_images,
            questions=questions
        )

    def preprocess_for_clip(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess images specifically for CLIP model.

        Args:
            images: List of PIL Images

        Returns:
            Tensor of preprocessed images ready for CLIP
        """
        processed = []
        for image in images:
            if image is not None:
                processed.append(self.image_transform(image))
            else:
                # Create placeholder
                placeholder = Image.new('RGB', self.image_size, color=(128, 128, 128))
                processed.append(self.image_transform(placeholder))

        return torch.stack(processed)

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate dataset integrity and return statistics.

        Returns:
            Dictionary with validation results and dataset statistics
        """
        validation_results = {
            "status": "success",
            "total_samples": self.total_samples,
            "total_images": 0,
            "missing_images": 0,
            "scenario_distribution": {},
            "errors": []
        }

        try:
            # Check images directory
            images_dir = self.data_path / "images"
            if images_dir.exists():
                image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                validation_results["total_images"] = len(image_files)
            else:
                validation_results["errors"].append("Images directory not found")

            # Validate samples
            scenario_counts = {scenario: 0 for scenario in self.perspective_types}
            missing_images = 0

            for question_data in self.questions_data:
                # Check perspective type mapping
                category = question_data.get('scenario', question_data.get('category', 'unknown'))
                perspective_type = self.category_to_perspective.get(category, 'angle')
                scenario_counts[perspective_type] += 1

                # Check image existence
                image_path = question_data.get('image_path', '')
                if image_path:
                    full_path = self.data_path / image_path
                    if not full_path.exists():
                        missing_images += 1

            validation_results["missing_images"] = missing_images
            validation_results["scenario_distribution"] = scenario_counts

            # Check if we have enough samples for each scenario
            min_samples_per_scenario = 10
            for scenario, count in scenario_counts.items():
                if count < min_samples_per_scenario:
                    validation_results["errors"].append(
                        f"Insufficient samples for scenario '{scenario}': {count} < {min_samples_per_scenario}"
                    )

            if validation_results["errors"]:
                validation_results["status"] = "warning"

            logger.info(f"Dataset validation complete: {validation_results['status']}")

        except Exception as e:
            validation_results["status"] = "error"
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            logger.error(f"Dataset validation failed: {e}")

        return validation_results

    def get_samples_by_scenario(self, scenario_type: str) -> List[Sample]:
        """
        Get all samples for a specific perspective change scenario.

        Args:
            scenario_type: One of ['angle', 'partial', 'scope', 'occlusion']

        Returns:
            List of Sample objects for the specified scenario
        """
        if scenario_type not in self.perspective_types:
            raise ValueError(f"Invalid scenario type: {scenario_type}. Must be one of {self.perspective_types}")

        samples = []
        for question_data in self.questions_data:
            # Map category to perspective type
            category = question_data.get('scenario', question_data.get('category', 'unknown'))
            perspective_type = self.category_to_perspective.get(category, 'angle')  # Default to angle

            if perspective_type == scenario_type:
                try:
                    sample = self._create_sample_from_question(question_data)
                    if sample:
                        samples.append(sample)
                except Exception as e:
                    logger.warning(f"Error creating sample for question {question_data.get('question_id', 'unknown')}: {e}")
                    continue

        logger.info(f"Found {len(samples)} samples for scenario: {scenario_type}")
        return samples

    def get_scenario_stats(self) -> Dict[str, int]:
        """
        Get sample count for each perspective change scenario.

        Returns:
            Dictionary mapping scenario types to sample counts
        """
        stats = {scenario: 0 for scenario in self.perspective_types}

        for question_data in self.questions_data:
            category = question_data.get('scenario', question_data.get('category', 'unknown'))
            perspective_type = self.category_to_perspective.get(category, 'angle')
            stats[perspective_type] += 1

        return stats

    def create_dataloader(self, scenario_type: str, shuffle: bool = False) -> DataLoader:
        """
        Create a PyTorch DataLoader for a specific scenario.

        Args:
            scenario_type: Perspective change scenario type
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader for the specified scenario
        """
        samples = list(self.load_scenario(scenario_type))

        class MRAGTorchDataset(Dataset):
            def __init__(self, samples_list, dataset_instance):
                self.samples = samples_list
                self.dataset = dataset_instance

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]

                # Load image if not loaded
                if sample.image is None and sample.image_path and os.path.exists(sample.image_path):
                    try:
                        sample.image = Image.open(sample.image_path).convert('RGB')
                    except Exception:
                        sample.image = Image.new('RGB', self.dataset.image_size, color=(128, 128, 128))

                # Preprocess for CLIP
                if sample.image:
                    image_tensor = self.dataset.image_transform(sample.image)
                else:
                    placeholder = Image.new('RGB', self.dataset.image_size, color=(128, 128, 128))
                    image_tensor = self.dataset.image_transform(placeholder)

                return {
                    'image': image_tensor,
                    'question': sample.question,
                    'question_id': sample.question_id,
                    'ground_truth': sample.ground_truth,
                    'perspective_type': sample.perspective_type,
                    'metadata': sample.metadata
                }

        torch_dataset = MRAGTorchDataset(samples, self)
        return DataLoader(
            torch_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues
            collate_fn=None
        )