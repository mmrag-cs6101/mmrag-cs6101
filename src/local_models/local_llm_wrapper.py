"""
Local LLM Wrapper for RAG-Anything Integration
Replaces OpenAI API calls with local open-source models
"""

import os
import torch
from typing import List, Dict, Any, Optional, Union
import logging
import gc
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    BitsAndBytesConfig
)
import asyncio
from functools import partial
import json

logger = logging.getLogger(__name__)


class LocalLLMWrapper:
    """
    Local LLM wrapper supporting multiple open-source models
    Compatible with RAG-Anything's model function signature
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_length: int = 1024,  # Reduced for faster generation
        temperature: float = 0.1,  # Ultra-low temperature for fastest, most deterministic responses
        top_p: float = 0.8,  # Slightly more focused sampling
        do_sample: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize local LLM wrapper
        
        Args:
            model_name: HuggingFace model name (e.g., "mistralai/Mistral-7B-Instruct-v0.1")
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
        
        # Initialize model and tokenizer
        self._load_model()
        
        logger.info(f"Local LLM initialized: {model_name} on {self.device}")
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": True
            }
            
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
            elif self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if not self.quantization_config and self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def _format_prompt(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format prompt for the model based on its expected format
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            history_messages: Conversation history
            
        Returns:
            Formatted prompt string
        """
        # Different models have different prompt formats
        if "mistral" in self.model_name.lower():
            return self._format_mistral_prompt(prompt, system_prompt, history_messages)
        elif "llama" in self.model_name.lower():
            return self._format_llama_prompt(prompt, system_prompt, history_messages)
        else:
            # Generic format
            return self._format_generic_prompt(prompt, system_prompt, history_messages)
    
    def _format_mistral_prompt(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format prompt for Mistral models"""
        formatted = ""
        
        if system_prompt:
            formatted += f"<s>[INST] {system_prompt}\n\n"
        else:
            formatted += "<s>[INST] "
        
        # Add history if present
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    formatted += f"{content}\n"
                elif role == "assistant":
                    formatted += f"[/INST] {content}</s><s>[INST] "
        
        formatted += f"{prompt} [/INST]"
        return formatted
    
    def _format_llama_prompt(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Format prompt for Llama models"""
        formatted = "<s>"
        
        if system_prompt:
            formatted += f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        
        # Add history if present
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    formatted += f"[INST] {content} [/INST] "
                elif role == "assistant":
                    formatted += f"{content}</s><s>"
        
        formatted += f"[INST] {prompt} [/INST]"
        return formatted
    
    def _format_generic_prompt(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generic prompt format"""
        formatted_parts = []
        
        if system_prompt:
            formatted_parts.append(f"System: {system_prompt}")
        
        # Add history if present
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted_parts.append(f"{role.capitalize()}: {content}")
        
        formatted_parts.append(f"User: {prompt}")
        formatted_parts.append("Assistant:")
        
        return "\n\n".join(formatted_parts)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Generate text response
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt
            history_messages: Conversation history
            max_length: Override max length
            temperature: Override temperature
            top_p: Override top_p
            do_sample: Override do_sample
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            # Format the full prompt
            full_prompt = self._format_prompt(prompt, system_prompt, history_messages)
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length // 2  # Reserve space for output
            ).to(self.device)
            
            # Generation parameters - filter out non-generation kwargs
            gen_kwargs = {
                "max_length": max_length or self.max_length,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "do_sample": do_sample if do_sample is not None else self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                # Ultra-fast generation settings
                "max_new_tokens": 64,  # Even more reduced for ultra-fast generation
                "use_cache": True,  # Enable KV cache for faster generation
                "num_beams": 1,  # Use greedy decoding for speed
                "do_sample": False,  # Disable sampling for deterministic, faster generation
                "early_stopping": True,
                "repetition_penalty": 1.0,  # Disable repetition penalty for speed
                "length_penalty": 1.0  # Disable length penalty for speed
            }
            
            # Only add valid generation parameters from kwargs
            valid_gen_params = {
                "max_new_tokens", "min_length", "min_new_tokens", 
                "num_beams", "num_beam_groups", "penalty_alpha",
                "use_cache", "typical_p", "epsilon_cutoff", "eta_cutoff",
                "diversity_penalty", "repetition_penalty", "encoder_repetition_penalty",
                "length_penalty", "no_repeat_ngram_size", "bad_words_ids",
                "force_words_ids", "renormalize_logits", "constraints",
                "forced_bos_token_id", "forced_eos_token_id", "remove_invalid_values",
                "exponential_decay_length_penalty", "suppress_tokens", 
                "begin_suppress_tokens", "forced_decoder_ids"
            }
            
            for key, value in kwargs.items():
                if key in valid_gen_params:
                    gen_kwargs[key] = value
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **gen_kwargs
                )
            
            # Decode only the generated part (excluding input)
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Remove any remaining special tokens
            for token in ["<s>", "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]:
                response = response.replace(token, "")
            
            response = response.strip()
            
            # Aggressive memory cleanup after generation
            del outputs, generated_tokens, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Cleanup on error as well
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return f"Error: Failed to generate response - {str(e)}"
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Async wrapper for generate method
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt
            history_messages: Conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        func = partial(
            self.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
        
        return await loop.run_in_executor(None, func)
    
    def get_rag_compatible_func(self):
        """
        Get RAG-Anything compatible function
        
        Returns:
            Function compatible with LightRAG's llm_model_func signature
        """
        async def rag_llm_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict[str, str]]] = None,
            **kwargs
        ) -> str:
            """RAG-compatible LLM function"""
            return await self.agenerate(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                **kwargs
            )
        
        return rag_llm_func


class LocalLLMManager:
    """Manager for multiple local LLM instances"""
    
    def __init__(self):
        self.models: Dict[str, LocalLLMWrapper] = {}
    
    def load_model(
        self,
        model_name: str,
        nickname: Optional[str] = None,
        **kwargs
    ) -> LocalLLMWrapper:
        """
        Load and cache a model
        
        Args:
            model_name: HuggingFace model name
            nickname: Optional nickname for the model
            **kwargs: Model initialization parameters
            
        Returns:
            LocalLLMWrapper instance
        """
        key = nickname or model_name
        
        if key not in self.models:
            self.models[key] = LocalLLMWrapper(model_name, **kwargs)
        
        return self.models[key]
    
    def get_model(self, key: str) -> Optional[LocalLLMWrapper]:
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


# Pre-configured model configurations
RECOMMENDED_MODELS = {
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",  # Updated to unrestricted version
        "load_in_4bit": True,
        "max_length": 4096
    },
    "llama2-7b": {
        "model_name": "NousResearch/Llama-2-7b-chat-hf",  # Community version (unrestricted)
        "load_in_4bit": True,
        "max_length": 4096
    },
    "codellama-7b": {
        "model_name": "codellama/CodeLlama-7b-Instruct-hf",
        "load_in_4bit": True,
        "max_length": 4096
    },
    "phi-3": {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "load_in_4bit": False,  # Phi-3 is small enough
        "max_length": 4096
    },
    "gemma-2b": {
        "model_name": "google/gemma-2b-it",  # Gemma 2B (unrestricted)
        "load_in_4bit": False,
        "max_length": 2048
    },
    "qwen-1.8b": {
        "model_name": "Qwen/Qwen2-1.5B-Instruct",  # Qwen (unrestricted and fast)
        "load_in_4bit": False,
        "max_length": 2048
    },
    "qwen-0.5b": {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",  # Ultra-fast tiny model for testing
        "load_in_4bit": False,
        "max_length": 1024
    }
}


def create_local_llm_func(
    model_key: str = "mistral-7b",
    custom_config: Optional[Dict[str, Any]] = None,
    manager: Optional[LocalLLMManager] = None
):
    """
    Create a local LLM function for RAG-Anything integration
    
    Args:
        model_key: Key for recommended model or custom model name
        custom_config: Custom model configuration
        manager: Optional model manager instance
        
    Returns:
        RAG-compatible LLM function
    """
    if manager is None:
        manager = LocalLLMManager()
    
    # Get model configuration
    if model_key in RECOMMENDED_MODELS:
        config = RECOMMENDED_MODELS[model_key].copy()
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
    
    async def test_local_llm():
        # Create local LLM function
        llm_func = create_local_llm_func("phi-3")  # Use Phi-3 as it's smaller
        
        # Test the function
        response = await llm_func(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful assistant."
        )
        
        print(f"Response: {response}")
    
    # Run test
    asyncio.run(test_local_llm())