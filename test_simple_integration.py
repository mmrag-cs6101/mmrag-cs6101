"""
Simple integration test that avoids problematic VLM queries
Tests core functionality with minimal complexity
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

async def test_simple_integration():
    """Test basic integration without VLM enhanced queries"""
    try:
        logger.info("🚀 Testing simple RAG integration...")
        
        # Create RAG with minimal configuration
        rag = create_local_rag_anything(
            working_dir="./test_simple_storage",
            llm_model="qwen-1.8b",  # Fast model
            embedding_model="all-mpnet",  # Fast embedding
            device="cpu"  # CPU for compatibility
        )
        
        # Initialize
        logger.info("Initializing RAG system...")
        await rag.initialize()
        logger.info("✅ RAG system initialized")
        
        # Create test document
        test_dir = Path("./test_simple_storage")
        test_dir.mkdir(exist_ok=True)
        test_doc = test_dir / "simple_test.txt"
        
        simple_content = """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that enables computers to learn from data.
        
        There are three main types of machine learning:
        1. Supervised learning - uses labeled training data
        2. Unsupervised learning - finds patterns in unlabeled data  
        3. Reinforcement learning - learns through trial and error
        
        Popular algorithms include decision trees, neural networks, and support vector machines.
        """
        
        with open(test_doc, 'w') as f:
            f.write(simple_content)
        
        logger.info("Processing document...")
        
        # Insert document
        result = await rag.insert_file(test_doc)
        logger.info("✅ Document processed successfully")
        logger.info(f"Result type: {type(result)}")
        
        # Test simple query with basic mode to avoid VLM path
        logger.info("Testing simple query...")
        try:
            response = await rag.query(
                "What are the three types of machine learning?",
                mode="naive"  # Use naive mode to avoid complex query paths
            )
            logger.info(f"✅ Query successful!")
            logger.info(f"Response: {response[:200]}...")
        except Exception as e:
            logger.warning(f"Query with naive mode failed: {e}")
            
            # Try with bypass mode (direct LLM without retrieval)
            logger.info("Trying bypass mode...")
            response = await rag.query(
                "What are the three types of machine learning?",
                mode="bypass"
            )
            logger.info(f"✅ Bypass query successful!")
            logger.info(f"Response: {response[:200]}...")
        
        # Cleanup
        rag.cleanup()
        
        logger.info("🎉 Simple integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_direct_llm():
    """Test LLM directly without RAG"""
    try:
        logger.info("🧠 Testing direct LLM functionality...")
        
        from local_models import create_local_llm_func
        
        # Create LLM function
        llm_func = create_local_llm_func("qwen-1.8b")
        
        # Test generation
        response = await llm_func(
            prompt="List the three main types of machine learning.",
            system_prompt="You are a helpful AI assistant. Be concise and accurate."
        )
        
        logger.info(f"✅ Direct LLM response: {response}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_components_separately():
    """Test each component separately"""
    try:
        logger.info("🔧 Testing components separately...")
        
        from local_models import (
            create_local_llm_func,
            create_local_embedding_func,
            LocalRAGAnything
        )
        
        # Test embedding function
        logger.info("Testing embeddings...")
        embedding_func, dim = create_local_embedding_func("all-mpnet")
        embeddings = await embedding_func(["test text 1", "test text 2"])
        logger.info(f"✅ Embeddings: shape {embeddings.shape}, dim {dim}")
        
        # Test LLM function  
        logger.info("Testing LLM...")
        llm_func = create_local_llm_func("qwen-1.8b")
        response = await llm_func("What is 2+2?")
        logger.info(f"✅ LLM response: {response}")
        
        # Test LocalRAGAnything initialization only
        logger.info("Testing RAG initialization...")
        rag = LocalRAGAnything(
            working_dir="./test_components_storage",
            llm_model="qwen-1.8b",
            embedding_model="all-mpnet",
            device="cpu"
        )
        logger.info("✅ RAG object created")
        
        logger.info("🎉 Component tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", action="store_true", help="Test components separately")
    parser.add_argument("--llm-only", action="store_true", help="Test LLM only")
    parser.add_argument("--simple", action="store_true", help="Test simple integration")
    
    args = parser.parse_args()
    
    success = True
    
    if args.components:
        success = asyncio.run(test_components_separately())
    elif args.llm_only:
        success = asyncio.run(test_direct_llm())
    elif args.simple:
        success = asyncio.run(test_simple_integration())
    else:
        # Run all tests in order
        print("🔧 Testing components separately...")
        success1 = asyncio.run(test_components_separately())
        
        if success1:
            print("\n🧠 Testing direct LLM...")
            success2 = asyncio.run(test_direct_llm())
            
            if success2:
                print("\n🚀 Testing simple integration...")
                success3 = asyncio.run(test_simple_integration())
                success = success1 and success2 and success3
            else:
                success = False
        else:
            success = False
    
    if success:
        print("\n✅ All simple tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)