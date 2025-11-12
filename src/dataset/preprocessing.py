"""
Data Preprocessing Pipeline

Advanced image preprocessing and data augmentation for MRAG-Bench dataset.
"""

import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
import numpy as np
from dataclasses import dataclass

try:
    from ..utils.memory_manager import MemoryManager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    image_size: Tuple[int, int] = (224, 224)
    clip_normalization: bool = True
    enable_augmentation: bool = False
    augmentation_probability: float = 0.3
    max_batch_size: int = 32
    memory_efficient: bool = True


class ImagePreprocessor:
    """
    Advanced image preprocessing pipeline with CLIP normalization and memory management.

    Supports:
    - CLIP-compatible preprocessing
    - Memory-efficient batch processing
    - Optional data augmentation
    - Perspective change scenario-specific preprocessing
    """

    def __init__(self, config: PreprocessingConfig, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize image preprocessor.

        Args:
            config: Preprocessing configuration
            memory_manager: Optional memory manager for optimization
        """
        self.config = config
        self.memory_manager = memory_manager or MemoryManager()

        # Setup transform pipelines
        self._setup_transforms()

        logger.info(f"ImagePreprocessor initialized with config: {config}")

    def _setup_transforms(self) -> None:
        """Setup image transformation pipelines."""
        # Base transforms
        base_transforms = [
            transforms.Resize(
                self.config.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(self.config.image_size)
        ]

        # CLIP normalization
        if self.config.clip_normalization:
            normalization = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalization = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        # Standard preprocessing pipeline
        self.standard_transform = transforms.Compose([
            *base_transforms,
            transforms.ToTensor(),
            normalization
        ])

        # Preprocessing without normalization (for debugging)
        self.basic_transform = transforms.Compose([
            *base_transforms,
            transforms.ToTensor()
        ])

        # Augmentation pipeline
        if self.config.enable_augmentation:
            augment_transforms = [
                *base_transforms,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.ToTensor(),
                normalization
            ]
            self.augmented_transform = transforms.Compose(augment_transforms)
        else:
            self.augmented_transform = self.standard_transform

    def preprocess_image(
        self,
        image: Union[Image.Image, str],
        apply_augmentation: bool = False,
        scenario_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image: PIL Image or path to image file
            apply_augmentation: Whether to apply data augmentation
            scenario_type: Perspective change scenario type for specialized preprocessing

        Returns:
            Preprocessed image tensor
        """
        # Load image if path provided
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {image}: {e}")
                image = self._create_placeholder_image()

        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply scenario-specific preprocessing
        if scenario_type:
            image = self._apply_scenario_preprocessing(image, scenario_type)

        # Apply transforms
        transform = self.augmented_transform if apply_augmentation else self.standard_transform

        try:
            return transform(image)
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            placeholder = self._create_placeholder_image()
            return transform(placeholder)

    def preprocess_batch(
        self,
        images: List[Union[Image.Image, str]],
        apply_augmentation: bool = False,
        scenario_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Preprocess a batch of images with memory management.

        Args:
            images: List of PIL Images or image paths
            apply_augmentation: Whether to apply data augmentation
            scenario_types: List of scenario types for each image

        Returns:
            Batch tensor of preprocessed images
        """
        if len(images) == 0:
            return torch.empty(0, 3, *self.config.image_size)

        # Memory management
        batch_size = min(len(images), self.config.max_batch_size)
        if self.memory_manager.check_memory_availability(batch_size * 0.1):  # Rough estimate
            batch_size = self.memory_manager.get_recommended_batch_size(
                batch_size, memory_per_item_mb=100
            )

        processed_tensors = []

        with self.memory_manager.memory_guard("image_preprocessing"):
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_scenarios = scenario_types[i:i + batch_size] if scenario_types else [None] * len(batch_images)

                batch_tensors = []
                for img, scenario in zip(batch_images, batch_scenarios):
                    tensor = self.preprocess_image(img, apply_augmentation, scenario)
                    batch_tensors.append(tensor)

                if batch_tensors:
                    processed_tensors.extend(batch_tensors)

                # Clear intermediate results
                if self.config.memory_efficient:
                    self.memory_manager.clear_gpu_memory()

        return torch.stack(processed_tensors)

    def _apply_scenario_preprocessing(self, image: Image.Image, scenario_type: str) -> Image.Image:
        """
        Apply scenario-specific preprocessing.

        Args:
            image: Input PIL image
            scenario_type: Type of perspective change scenario

        Returns:
            Preprocessed image
        """
        try:
            if scenario_type == 'angle':
                # For angle changes, ensure good contrast and sharpness
                image = self._enhance_contrast(image, factor=1.1)
                image = self._enhance_sharpness(image, factor=1.05)

            elif scenario_type == 'partial':
                # For partial views, enhance edge detection
                image = self._enhance_sharpness(image, factor=1.1)

            elif scenario_type == 'scope':
                # For scope variations, normalize brightness
                image = self._normalize_brightness(image)

            elif scenario_type == 'occlusion':
                # For occlusions, enhance contrast to make visible parts clearer
                image = self._enhance_contrast(image, factor=1.15)

        except Exception as e:
            logger.warning(f"Failed to apply scenario preprocessing for {scenario_type}: {e}")

        return image

    def _enhance_contrast(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """Enhance image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def _enhance_sharpness(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """Enhance image sharpness."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def _normalize_brightness(self, image: Image.Image) -> Image.Image:
        """Normalize image brightness."""
        # Convert to array for processing
        img_array = np.array(image)

        # Calculate current brightness
        brightness = np.mean(img_array)
        target_brightness = 128  # Target middle brightness

        # Adjust brightness
        if brightness > 0:
            adjustment_factor = target_brightness / brightness
            adjustment_factor = np.clip(adjustment_factor, 0.8, 1.2)  # Limit adjustment

            img_array = img_array * adjustment_factor
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def _create_placeholder_image(self) -> Image.Image:
        """Create a placeholder image."""
        return Image.new('RGB', self.config.image_size, color=(128, 128, 128))

    def get_transform_for_clip(self) -> transforms.Compose:
        """Get the transform pipeline specifically for CLIP model."""
        return self.standard_transform

    def get_basic_transform(self) -> transforms.Compose:
        """Get basic transform without normalization."""
        return self.basic_transform


class BatchDataProcessor:
    """
    Efficient batch data processing with memory management.

    Handles:
    - Batch creation and optimization
    - Memory-efficient processing
    - Dynamic batch size adjustment
    """

    def __init__(self, preprocessor: ImagePreprocessor, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize batch processor.

        Args:
            preprocessor: Image preprocessor instance
            memory_manager: Optional memory manager
        """
        self.preprocessor = preprocessor
        self.memory_manager = memory_manager or MemoryManager()

    def process_batch(
        self,
        batch_data: Dict[str, Any],
        target_device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of data for model input.

        Args:
            batch_data: Dictionary containing images, questions, etc.
            target_device: Target device for tensors

        Returns:
            Dictionary of processed tensors
        """
        processed_batch = {}

        with self.memory_manager.memory_guard("batch_processing"):
            # Process images
            if 'images' in batch_data:
                images = batch_data['images']
                scenario_types = batch_data.get('scenario_types', None)

                image_tensors = self.preprocessor.preprocess_batch(
                    images, scenario_types=scenario_types
                )

                # Move to target device if available
                if torch.cuda.is_available() and target_device == "cuda":
                    image_tensors = image_tensors.to(target_device)

                processed_batch['images'] = image_tensors

            # Process other data
            for key, value in batch_data.items():
                if key not in ['images', 'scenario_types']:
                    if isinstance(value, torch.Tensor):
                        if torch.cuda.is_available() and target_device == "cuda":
                            processed_batch[key] = value.to(target_device)
                        else:
                            processed_batch[key] = value
                    else:
                        processed_batch[key] = value

        return processed_batch

    def get_optimal_batch_size(self, base_batch_size: int, image_size: Tuple[int, int]) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            base_batch_size: Desired batch size
            image_size: Size of images (width, height)

        Returns:
            Optimal batch size
        """
        # Estimate memory per image (rough calculation)
        pixels_per_image = image_size[0] * image_size[1] * 3  # RGB
        bytes_per_image = pixels_per_image * 4  # float32
        mb_per_image = bytes_per_image / (1024 * 1024)

        # Add overhead for preprocessing
        mb_per_image *= 2  # Double for processing overhead

        return self.memory_manager.get_recommended_batch_size(
            base_batch_size, mb_per_image
        )