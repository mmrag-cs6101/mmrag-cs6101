"""
MRAG-Bench HuggingFace Dataset Wrapper

Loads the official MRAG-Bench dataset from HuggingFace with proper multiple-choice format.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class MRAGSample:
    """Sample from MRAG-Bench dataset."""
    question_id: str
    question: str
    choices: Dict[str, str]  # {"A": "answer1", "B": "answer2", ...}
    answer_choice: str  # "A", "B", "C", or "D"
    answer: str  # The actual answer text
    query_image: Image.Image
    gt_images: List[Image.Image]
    retrieved_images: List[Image.Image]
    scenario: str
    aspect: str
    image_type: str


class MRAGHFDataset:
    """
    Wrapper for official MRAG-Bench dataset from HuggingFace.

    Loads uclanlp/MRAG-Bench with proper multiple-choice format and images.
    """

    def __init__(self, split: str = "test", use_retrieved: bool = False):
        """
        Initialize MRAG-Bench dataset from HuggingFace.

        Args:
            split: Dataset split ("test" by default)
            use_retrieved: If True, use retrieved_images; if False, use gt_images
        """
        logger.info(f"Loading MRAG-Bench dataset from HuggingFace (split={split})...")
        self.dataset = load_dataset("uclanlp/MRAG-Bench", split=split)
        self.use_retrieved = use_retrieved
        logger.info(f"Loaded {len(self.dataset)} samples")

        # Map scenario names to perspective change types
        self.scenario_mapping = {
            'Scope': 'scope',
            'Obstruction': 'occlusion',  # Fixed: Obstruction maps to occlusion, not scope
            'Temporal': 'partial',
            'Deformation': 'scope',
            'Biological': 'angle',
            'Angle': 'angle',
            'Partial': 'partial',
            'Incomplete': 'partial',
            'Others': 'scope'
        }

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> MRAGSample:
        """Get sample by index."""
        raw_sample = self.dataset[idx]
        return self._parse_sample(raw_sample, idx)

    def _parse_sample(self, raw_sample: Dict[str, Any], idx: int) -> MRAGSample:
        """Parse raw HuggingFace sample into MRAGSample."""
        # Build choices dict
        choices = {
            "A": raw_sample["A"],
            "B": raw_sample["B"],
            "C": raw_sample["C"],
            "D": raw_sample["D"]
        }

        # Use retrieved images or ground-truth images
        context_images = (
            raw_sample["retrieved_images"] if self.use_retrieved
            else raw_sample["gt_images"]
        )

        return MRAGSample(
            question_id=f"mrag_{idx:06d}",
            question=raw_sample["question"],
            choices=choices,
            answer_choice=raw_sample["answer_choice"],
            answer=raw_sample["answer"],
            query_image=raw_sample["image"],
            gt_images=raw_sample["gt_images"],
            retrieved_images=raw_sample["retrieved_images"],
            scenario=raw_sample["scenario"],
            aspect=raw_sample.get("aspect", ""),
            image_type=raw_sample.get("image_type", "")
        )

    def get_samples_by_scenario(self, scenario: str, max_samples: Optional[int] = None) -> List[MRAGSample]:
        """
        Get all samples for a specific perspective change scenario.

        Args:
            scenario: Scenario type ('angle', 'partial', 'scope', 'occlusion')
            max_samples: Maximum number of samples to return (None for all)

        Returns:
            List of MRAGSample objects
        """
        samples = []

        for idx, raw_sample in enumerate(self.dataset):
            # Map raw scenario to perspective type
            raw_scenario = raw_sample["scenario"]
            perspective_type = self.scenario_mapping.get(raw_scenario, "scope")

            if perspective_type == scenario:
                sample = self._parse_sample(raw_sample, idx)
                samples.append(sample)

                if max_samples and len(samples) >= max_samples:
                    break

        logger.info(f"Found {len(samples)} samples for scenario: {scenario}")
        return samples

    def get_all_samples(self, max_samples: Optional[int] = None) -> List[MRAGSample]:
        """Get all samples."""
        samples = []
        for idx in range(len(self.dataset)):
            samples.append(self[idx])
            if max_samples and len(samples) >= max_samples:
                break
        return samples
