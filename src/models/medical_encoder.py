"""
Medical Image Encoder using CLIP for medical multimodal RAG
Handles encoding of medical images with domain-specific preprocessing
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import Union, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MedicalImageEncoder:
    """
    Medical image encoder based on CLIP with medical domain adaptations
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize medical image encoder
        
        Args:
            model_name: CLIP model to use
            device: Device to run model on
            cache_dir: Directory to cache model files
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Medical image preprocessing parameters
        self.medical_transforms = self._setup_medical_transforms()
        
        logger.info(f"Medical image encoder initialized on {self.device}")
    
    def _setup_medical_transforms(self):
        """Setup medical-specific image preprocessing"""
        return {
            'chest_xray': {
                'window_center': 127.5,
                'window_width': 255,
                'invert': False
            },
            'ct_scan': {
                'window_center': 40,  # Soft tissue window
                'window_width': 400,
                'invert': False
            },
            'histology': {
                'normalize': True,
                'enhance_contrast': True
            }
        }
    
    def preprocess_medical_image(
        self, 
        image: Union[str, Image.Image, np.ndarray], 
        modality: str = 'general'
    ) -> Image.Image:
        """
        Preprocess medical image based on modality
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            modality: Medical imaging modality ('chest_xray', 'ct_scan', 'histology', 'general')
            
        Returns:
            Preprocessed PIL Image
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply modality-specific preprocessing
        if modality in self.medical_transforms:
            transforms = self.medical_transforms[modality]
            
            if modality == 'chest_xray':
                # Apply windowing for X-rays
                image_array = np.array(image)
                if not transforms['invert']:
                    # Normalize to standard range
                    image_array = np.clip(image_array, 0, 255)
                image = Image.fromarray(image_array.astype(np.uint8))
                
            elif modality == 'histology':
                # Enhance contrast for histology images
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
        
        return image
    
    def encode_image(
        self, 
        image: Union[str, Image.Image, np.ndarray], 
        modality: str = 'general',
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode single medical image to embedding vector
        
        Args:
            image: Input image
            modality: Medical imaging modality
            normalize: Whether to L2 normalize the embedding
            
        Returns:
            Image embedding as numpy array
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_medical_image(image, modality)
            
            # Process with CLIP
            inputs = self.processor(images=processed_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
                if normalize:
                    # L2 normalize for cosine similarity
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def encode_images_batch(
        self, 
        images: List[Union[str, Image.Image]], 
        modalities: Optional[List[str]] = None,
        batch_size: int = 8,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode multiple images in batches
        
        Args:
            images: List of images to encode
            modalities: List of modalities for each image
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Array of image embeddings [num_images, embedding_dim]
        """
        if modalities is None:
            modalities = ['general'] * len(images)
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_modalities = modalities[i:i + batch_size]
            
            # Process batch
            processed_images = []
            for img, mod in zip(batch_images, batch_modalities):
                processed_images.append(self.preprocess_medical_image(img, mod))
            
            # Encode batch
            inputs = self.processor(images=processed_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
                if normalize:
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text queries for image retrieval
        
        Args:
            text: Text query or list of queries
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Text embedding(s) as numpy array
        """
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
                if normalize:
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            embeddings = text_features.cpu().numpy()
            return embeddings[0] if isinstance(text, str) else embeddings
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def compute_similarity(
        self, 
        image_embeddings: np.ndarray, 
        text_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings
        
        Args:
            image_embeddings: Array of image embeddings [num_images, embed_dim]
            text_embedding: Text embedding [embed_dim]
            
        Returns:
            Similarity scores [num_images]
        """
        # Ensure 2D arrays
        if len(image_embeddings.shape) == 1:
            image_embeddings = image_embeddings.reshape(1, -1)
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = np.dot(image_embeddings, text_embedding.T).flatten()
        return similarities
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings"""
        return self.model.config.projection_dim
    
    def save_model(self, path: str):
        """Save the model to disk"""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None):
        """Load model from disk"""
        instance = cls.__new__(cls)
        instance.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        instance.model = CLIPModel.from_pretrained(path)
        instance.processor = CLIPProcessor.from_pretrained(path)
        instance.model.to(instance.device)
        instance.model.eval()
        
        instance.medical_transforms = instance._setup_medical_transforms()
        
        logger.info(f"Model loaded from {path}")
        return instance


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize encoder
    encoder = MedicalImageEncoder()
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    
    # Test text encoding
    medical_queries = [
        "chest x-ray showing pneumonia",
        "brain MRI with tumor",
        "skin lesion dermatology image"
    ]
    
    text_embeddings = encoder.encode_text(medical_queries)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Test with dummy image
    dummy_image = Image.new('RGB', (224, 224), color='white')
    image_embedding = encoder.encode_image(dummy_image, modality='chest_xray')
    print(f"Image embedding shape: {image_embedding.shape}")
    
    # Test similarity
    similarities = encoder.compute_similarity(
        image_embedding.reshape(1, -1), 
        text_embeddings[0].reshape(1, -1)
    )
    print(f"Similarity score: {similarities[0]:.4f}")