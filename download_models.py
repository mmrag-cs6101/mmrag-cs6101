"""
Model Pre-download Script
Downloads and caches models before running RAG-Anything
"""

import os
import sys
from pathlib import Path
from typing import Optional
import logging

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_hf_model(model_name: str, cache_dir: Optional[str] = None, use_auth_token: bool = False):
    """Download a HuggingFace model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Downloading model: {model_name}")
        
        # Set authentication token if needed
        token = True if use_auth_token else None
        
        # Download tokenizer
        logger.info("  - Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=token
        )
        
        # Download model
        logger.info("  - Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=token,
            torch_dtype="auto"  # Don't load into memory
        )
        
        logger.info(f"✅ Successfully downloaded: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        return False

def download_sentence_transformer(model_name: str, cache_dir: Optional[str] = None):
    """Download a sentence-transformer model"""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Downloading sentence-transformer: {model_name}")
        
        model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir
        )
        
        logger.info(f"✅ Successfully downloaded: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        return False

def download_vision_model(model_name: str, cache_dir: Optional[str] = None):
    """Download a vision model"""
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        logger.info(f"Downloading vision model: {model_name}")
        
        # Download processor
        logger.info("  - Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Download model
        logger.info("  - Downloading vision model weights...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype="auto"  # Don't load into memory
        )
        
        logger.info(f"✅ Successfully downloaded: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        return False

def main():
    """Main download function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-download models for Local RAG-Anything")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory")
    parser.add_argument("--preset", choices=["fast", "balanced", "quality", "medical", "all"], 
                       default="balanced", help="Model preset to download")
    parser.add_argument("--auth", action="store_true", help="Use HuggingFace authentication token")
    parser.add_argument("--models-only", action="store_true", help="Download only LLM models")
    parser.add_argument("--custom-model", type=str, help="Download a specific model")
    
    args = parser.parse_args()
    
    # Model configurations matching our presets
    presets = {
        "fast": {
            "llm": "Qwen/Qwen2-1.5B-Instruct",
            "vision": "llava-hf/llava-1.5-7b-hf",
            "embedding": "sentence-transformers/all-mpnet-base-v2"
        },
        "balanced": {
            "llm": "google/gemma-2b-it",
            "vision": "llava-hf/llava-1.5-7b-hf", 
            "embedding": "BAAI/bge-base-en-v1.5"
        },
        "quality": {
            "llm": "mistralai/Mistral-7B-Instruct-v0.3",
            "vision": "llava-hf/llava-1.5-7b-hf",
            "embedding": "BAAI/bge-large-en-v1.5"
        },
        "medical": {
            "llm": "google/gemma-2b-it",
            "vision": "llava-hf/llava-1.5-7b-hf",
            "embedding": "BAAI/bge-base-en-v1.5"
        }
    }
    
    # Additional popular models
    popular_models = {
        "mistral-gated": "mistralai/Mistral-7B-Instruct-v0.1",  # Requires auth
        "llama2": "NousResearch/Llama-2-7b-chat-hf",
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "moondream": "vikhyatk/moondream2"
    }
    
    success_count = 0
    total_count = 0
    
    if args.custom_model:
        # Download specific model
        logger.info(f"Downloading custom model: {args.custom_model}")
        if download_hf_model(args.custom_model, args.cache_dir, args.auth):
            success_count += 1
        total_count += 1
        
    elif args.preset == "all":
        # Download all presets
        logger.info("Downloading all model presets...")
        
        for preset_name, models in presets.items():
            logger.info(f"\n📦 Downloading {preset_name} preset models...")
            
            # LLM
            if download_hf_model(models["llm"], args.cache_dir, args.auth):
                success_count += 1
            total_count += 1
            
            if not args.models_only:
                # Vision model
                if download_vision_model(models["vision"], args.cache_dir):
                    success_count += 1
                total_count += 1
                
                # Embedding model
                if download_sentence_transformer(models["embedding"], args.cache_dir):
                    success_count += 1
                total_count += 1
        
        # Download popular additional models
        logger.info(f"\n📦 Downloading popular additional models...")
        for model_name, model_id in popular_models.items():
            if model_name == "mistral-gated" and not args.auth:
                logger.info(f"Skipping {model_id} (requires auth token)")
                continue
                
            if download_hf_model(model_id, args.cache_dir, args.auth):
                success_count += 1
            total_count += 1
        
    else:
        # Download specific preset
        models = presets[args.preset]
        logger.info(f"📦 Downloading {args.preset} preset models...")
        
        # LLM
        if download_hf_model(models["llm"], args.cache_dir, args.auth):
            success_count += 1
        total_count += 1
        
        if not args.models_only:
            # Vision model
            if download_vision_model(models["vision"], args.cache_dir):
                success_count += 1
            total_count += 1
            
            # Embedding model
            if download_sentence_transformer(models["embedding"], args.cache_dir):
                success_count += 1
            total_count += 1
    
    # Summary
    logger.info(f"\n📊 Download Summary:")
    logger.info(f"✅ Successful: {success_count}/{total_count}")
    
    if args.cache_dir:
        logger.info(f"📁 Models saved to: {args.cache_dir}")
    else:
        logger.info(f"📁 Models saved to default HuggingFace cache")
    
    if success_count == total_count:
        logger.info("🎉 All downloads completed successfully!")
        return 0
    else:
        logger.error("❌ Some downloads failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())