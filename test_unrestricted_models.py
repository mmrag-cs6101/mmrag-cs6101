"""
Test with unrestricted models only - no gated repos
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from local_models import create_local_rag_anything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_unrestricted_models():
    """Test with fastest unrestricted models"""
    try:
        logger.info("🚀 Testing with unrestricted models...")
        
        # Use only text-based testing for speed
        rag = create_local_rag_anything(
            working_dir="./test_unrestricted_storage",
            llm_model="qwen-1.8b",  # Fast unrestricted model
            # Skip vision model for faster testing
            embedding_model="all-mpnet",  # Fast embedding model
            device="cpu"  # Use CPU for broader compatibility
        )
        
        # Initialize
        logger.info("Initializing with unrestricted models...")
        await rag.initialize()
        logger.info("✅ Initialization successful")
        
        # Create test document
        test_dir = Path("./test_unrestricted_storage")
        test_dir.mkdir(exist_ok=True)
        test_doc = test_dir / "test_document.txt"
        
        test_content = """
        Machine Learning and Artificial Intelligence
        
        Machine learning is a method of data analysis that automates analytical model building.
        It is a branch of artificial intelligence based on the idea that systems can learn from data,
        identify patterns and make decisions with minimal human intervention.
        
        Types of machine learning include:
        1. Supervised learning
        2. Unsupervised learning  
        3. Reinforcement learning
        """
        
        with open(test_doc, 'w') as f:
            f.write(test_content)
        
        logger.info("Processing test document...")
        
        # Insert document
        result = await rag.insert_file(test_doc)
        logger.info("✅ Document processed successfully")
        
        # Test basic query
        logger.info("Testing basic query...")
        response = await rag.query("What types of machine learning are mentioned?")
        logger.info(f"✅ Query response: {response}")
        
        # Test another query
        logger.info("Testing second query...")
        response2 = await rag.query("What is machine learning?")
        logger.info(f"✅ Second query response: {response2}")
        
        # Cleanup
        rag.cleanup()
        
        logger.info("🎉 All tests passed with unrestricted models!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_loading_only():
    """Test just model loading without full RAG"""
    try:
        logger.info("🔍 Testing individual model loading...")
        
        from local_models import create_local_llm_func, create_local_embedding_func
        
        # Test embedding model (fastest)
        logger.info("Loading embedding model...")
        embedding_func, dim = create_local_embedding_func("all-mpnet")
        logger.info(f"✅ Embedding model loaded: dimension {dim}")
        
        # Test embeddings
        embeddings = await embedding_func(["test text", "another test"])
        logger.info(f"✅ Embeddings generated: shape {embeddings.shape}")
        
        # Test LLM model
        logger.info("Loading LLM model...")
        llm_func = create_local_llm_func("qwen-1.8b")
        logger.info("✅ LLM model loaded")
        
        # Test text generation
        response = await llm_func("What is 2+2?", system_prompt="Be concise.")
        logger.info(f"✅ LLM response: {response}")
        
        logger.info("🎉 Model loading tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-only", action="store_true", help="Test model loading only")
    parser.add_argument("--full", action="store_true", help="Test full RAG integration")
    
    args = parser.parse_args()
    
    if args.models_only:
        success = asyncio.run(test_model_loading_only())
    elif args.full:
        success = asyncio.run(test_unrestricted_models())
    else:
        # Run both tests
        print("Running model loading test first...")
        success1 = asyncio.run(test_model_loading_only())
        
        if success1:
            print("\nRunning full integration test...")
            success2 = asyncio.run(test_unrestricted_models())
            success = success1 and success2
        else:
            success = False
    
    if success:
        print("\n✅ All tests successful with unrestricted models!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)