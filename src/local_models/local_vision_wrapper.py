"""
Local Vision Model Wrapper for RAG-Anything Integration
Replaces OpenAI Vision API calls with local multimodal models
"""

import os
import torch
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
from PIL import Image
import base64
import io
import asyncio
from functools import partial

# Vision model imports
try:
    from transformers import (
        LlavaNextProcessor, LlavaNextForConditionalGeneration,
        AutoProcessor, AutoModelForVision2Seq,
        BitsAndBytesConfig
    )
except ImportError:
    logger.warning("Some vision model dependencies not available. Install with: pip install transformers[vision]")

logger = logging.getLogger(__name__)


class LocalVisionWrapper:
    """
    Local Vision Model wrapper supporting multiple open-source multimodal models
    Compatible with RAG-Anything's vision model function signature
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize local vision model wrapper
        
        Args:
            model_name: HuggingFace model name (e.g., "llava-hf/llava-1.5-7b-hf")
            device: Device to use ("cuda", "cpu", or "auto")
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization  
            max_length: Maximum generation length
            temperature: Generation temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            cache_dir: Model cache directory
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.cache_dir = cache_dir
        
        # Setup quantization config if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.quantization_config = quantization_config
        
        # Initialize model and processor
        self._load_model()
        
        logger.info(f"Local Vision Model initialized: {model_name} on {self.device}")
    
    def _load_model(self):
        """Load vision model and processor"""
        try:
            # Model loading arguments
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": True
            }
            
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
            elif self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            # Load model based on type
            if "llava" in self.model_name.lower():
                self.processor = LlavaNextProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                # Try generic vision-to-seq models
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            
            if not self.quantization_config and self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"Successfully loaded vision model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading vision model {self.model_name}: {e}")
            raise
    
    def _decode_base64_image(self, image_data: str) -> Image.Image:
        """
        Decode base64 image data
        
        Args:
            image_data: Base64 encoded image
            
        Returns:
            PIL Image object
        """
        try:
            # Remove data URL prefix if present
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise ValueError(f"Failed to decode image data: {str(e)}")
    
    def _prepare_messages_input(self, messages: List[Dict[str, Any]]) -> tuple:
        """
        Prepare messages input for multimodal processing
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Tuple of (text_prompt, images)
        """
        text_parts = []
        images = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                text_parts.append(f"System: {content}")
            elif isinstance(content, str):
                text_parts.append(f"{role.capitalize()}: {content}")
            elif isinstance(content, list):
                # Handle multimodal content
                text_content = []
                for item in content:
                    if item.get("type") == "text":
                        text_content.append(item["text"])
                    elif item.get("type") == "image_url":
                        # Extract image data
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",")[1]
                            image = self._decode_base64_image(image_data)
                            images.append(image)
                        else:
                            logger.warning(f"Unsupported image URL format: {image_url}")
                
                if text_content:
                    text_parts.append(f"{role.capitalize()}: {' '.join(text_content)}")
        
        prompt = "\n\n".join(text_parts)
        return prompt, images
    
    def generate_vision_response(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Generate vision-language response
        
        Args:
            prompt: Text prompt
            image_data: Base64 encoded image data
            system_prompt: System prompt
            history_messages: Conversation history
            messages: Multimodal messages (alternative to other params)
            max_length: Override max length
            temperature: Override temperature
            top_p: Override top_p
            do_sample: Override do_sample
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            images = []
            text_prompt = prompt
            
            # Handle different input formats
            if messages:
                # Use messages format (multimodal)
                text_prompt, images = self._prepare_messages_input(messages)
            else:
                # Traditional format
                if system_prompt:
                    text_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                
                if image_data:
                    image = self._decode_base64_image(image_data)
                    images = [image]
            
            # Handle case with no images (fallback to text-only)
            if not images:
                logger.warning("No images provided to vision model, generating text-only response")
                return self._generate_text_only_response(text_prompt, **kwargs)
            
            # Prepare inputs for vision model
            if "llava" in self.model_name.lower():
                # LLaVA format
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {"type": "image"}
                        ]
                    }
                ]
                
                prompt_text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True
                )
                
                inputs = self.processor(
                    images=images[0] if len(images) == 1 else images,
                    text=prompt_text,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Generic vision model format
                inputs = self.processor(
                    images=images[0] if len(images) == 1 else images,
                    text=text_prompt,
                    return_tensors="pt"
                ).to(self.device)
            
            # Generation parameters
            gen_kwargs = {
                "max_length": max_length or self.max_length,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "do_sample": do_sample if do_sample is not None else self.do_sample,
                "early_stopping": True
            }
            gen_kwargs.update(kwargs)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Decode response
            if "llava" in self.model_name.lower():
                # For LLaVA, decode only the generated part
                generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                response = self.processor.decode(generated_tokens, skip_special_tokens=True)
            else:
                # For other models, decode the full output
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Remove any prompt echoing (common in some models)
            if text_prompt in response:
                response = response.replace(text_prompt, "").strip()
            
            # Remove common artifacts
            for artifact in ["<s>", "</s>", "[INST]", "[/INST]", "System:", "User:", "Assistant:"]:
                response = response.replace(artifact, "")
            
            response = response.strip()
            
            # Ensure we have a valid response
            if not response:
                return "I can see the image, but I'm unable to provide a specific description at this time."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating vision response: {e}")
            return f"Error: Failed to process image and generate response - {str(e)}"
    
    def _generate_text_only_response(self, prompt: str, **kwargs) -> str:
        """Fallback text-only generation when no images are provided"""
        try:
            # Use the text generation capabilities if available
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
            
            gen_kwargs = {
                "max_length": kwargs.get("max_length", self.max_length),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "do_sample": kwargs.get("do_sample", self.do_sample),
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response or "I can help you with text-based questions."
            
        except Exception as e:
            logger.warning(f"Text-only fallback failed: {e}")
            return "I specialize in vision-language tasks. Please provide an image along with your question."
    
    async def agenerate_vision_response(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """
        Async wrapper for generate_vision_response
        
        Args:
            prompt: Text prompt
            image_data: Base64 encoded image data
            system_prompt: System prompt
            history_messages: Conversation history
            messages: Multimodal messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        func = partial(
            self.generate_vision_response,
            prompt=prompt,
            image_data=image_data,
            system_prompt=system_prompt,
            history_messages=history_messages,
            messages=messages,
            **kwargs
        )
        
        return await loop.run_in_executor(None, func)
    
    def get_rag_compatible_func(self):
        """
        Get RAG-Anything compatible vision function
        
        Returns:
            Function compatible with RAG-Anything's vision_model_func signature
        """
        async def rag_vision_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict[str, str]]] = None,
            image_data: Optional[str] = None,
            messages: Optional[List[Dict[str, Any]]] = None,
            **kwargs
        ) -> str:
            """RAG-compatible vision function"""
            return await self.agenerate_vision_response(
                prompt=prompt,
                image_data=image_data,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                messages=messages,
                **kwargs
            )
        
        return rag_vision_func


class LocalVisionManager:
    """Manager for multiple local vision model instances"""
    
    def __init__(self):
        self.models: Dict[str, LocalVisionWrapper] = {}
    
    def load_model(
        self,
        model_name: str,
        nickname: Optional[str] = None,
        **kwargs
    ) -> LocalVisionWrapper:
        """
        Load and cache a vision model
        
        Args:
            model_name: HuggingFace model name
            nickname: Optional nickname for the model
            **kwargs: Model initialization parameters
            
        Returns:
            LocalVisionWrapper instance
        """
        key = nickname or model_name
        
        if key not in self.models:
            self.models[key] = LocalVisionWrapper(model_name, **kwargs)
        
        return self.models[key]
    
    def get_model(self, key: str) -> Optional[LocalVisionWrapper]:
        """Get model by key"""
        return self.models.get(key)
    
    def list_models(self) -> List[str]:
        """List loaded models"""
        return list(self.models.keys())
    
    def unload_model(self, key: str):
        """Unload model to free memory"""
        if key in self.models:
            del self.models[key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Pre-configured vision model configurations
RECOMMENDED_VISION_MODELS = {
    "llava-1.5-7b": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "load_in_4bit": True,
        "max_length": 2048
    },
    "llava-1.5-13b": {
        "model_name": "llava-hf/llava-1.5-13b-hf",
        "load_in_4bit": True,
        "max_length": 2048
    },
    "internvl2": {
        "model_name": "OpenGVLab/InternVL2-2B",
        "load_in_4bit": False,
        "max_length": 2048
    }
}


def create_local_vision_func(
    model_key: str = "llava-1.5-7b",
    custom_config: Optional[Dict[str, Any]] = None,
    manager: Optional[LocalVisionManager] = None
):
    """
    Create a local vision model function for RAG-Anything integration
    
    Args:
        model_key: Key for recommended model or custom model name
        custom_config: Custom model configuration
        manager: Optional model manager instance
        
    Returns:
        RAG-compatible vision function
    """
    if manager is None:
        manager = LocalVisionManager()
    
    # Get model configuration
    if model_key in RECOMMENDED_VISION_MODELS:
        config = RECOMMENDED_VISION_MODELS[model_key].copy()
    else:
        config = {"model_name": model_key}
    
    if custom_config:
        config.update(custom_config)
    
    # Load model
    model = manager.load_model(nickname=model_key, **config)
    
    return model.get_rag_compatible_func()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_local_vision():
        # Create local vision function
        vision_func = create_local_vision_func("llava-1.5-7b")
        
        # Test with a simple prompt (no image)
        response = await vision_func(
            prompt="What can you help me with?",
            system_prompt="You are a helpful vision assistant."
        )
        
        print(f"Response: {response}")
    
    # Run test
    asyncio.run(test_local_vision())