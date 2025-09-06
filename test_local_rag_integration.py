"""
Test Local RAG-Anything Integration
Verifies that the local models work correctly with RAG-Anything
"""

import os
import sys
import logging
import asyncio
import pytest
from pathlib import Path
from typing import Optional

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import our local models
try:
    from local_models import (
        create_local_llm_func,
        create_local_vision_func, 
        create_local_embedding_func,
        create_local_rag_anything
    )
except ImportError as e:
    print(f"Failed to import local models: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLocalModels:
    """Test suite for local model wrappers"""
    
    @pytest.mark.asyncio
    async def test_local_llm_wrapper(self):
        """Test local LLM wrapper"""
        logger.info("Testing Local LLM Wrapper...")
        
        try:
            # Create LLM function (use smallest available model for testing)
            llm_func = create_local_llm_func("phi-3")
            
            # Test basic text generation
            response = await llm_func(
                prompt="What is the capital of France?",
                system_prompt="You are a helpful assistant. Give short, accurate answers."
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            logger.info(f"✅ LLM response: {response[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ LLM test failed: {e}")
            raise
    
    @pytest.mark.asyncio 
    async def test_local_embedding_wrapper(self):
        """Test local embedding wrapper"""
        logger.info("Testing Local Embedding Wrapper...")
        
        try:
            # Create embedding function
            embedding_func, embedding_dim = create_local_embedding_func("bge-small")
            
            # Test embedding generation
            texts = [
                "Machine learning is a subset of artificial intelligence",
                "Deep learning uses neural networks",
                "Natural language processing handles text data"
            ]
            
            embeddings = await embedding_func(texts)
            
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] == embedding_dim
            logger.info(f"✅ Embeddings shape: {embeddings.shape}, dim: {embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Embedding test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_local_vision_wrapper(self):
        """Test local vision wrapper"""
        logger.info("Testing Local Vision Wrapper...")
        
        try:
            # Create vision function
            vision_func = create_local_vision_func("llava-1.5-7b")
            
            # Test text-only fallback (no image provided)
            response = await vision_func(
                prompt="What can you help me with?",
                system_prompt="You are a helpful vision assistant."
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            logger.info(f"✅ Vision response (text-only): {response[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Vision test failed: {e}")
            raise


class TestRAGIntegration:
    """Test suite for RAG-Anything integration"""
    
    def setup_method(self):
        """Setup for each test"""
        self.test_dir = Path("./test_rag_storage")
        self.test_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_basic_rag_setup(self):
        """Test basic RAG system setup"""
        logger.info("Testing Basic RAG Setup...")
        
        try:
            # Create RAG instance with fastest preset
            rag = create_local_rag_anything(
                working_dir=str(self.test_dir),
                model_preset="fast"
            )
            
            # Initialize
            await rag.initialize()
            
            # Check model info
            info = rag.get_model_info()
            assert "llm_model" in info
            assert "embedding_dim" in info
            logger.info(f"✅ RAG setup successful: {info}")
            
            # Cleanup
            rag.cleanup()
            
        except Exception as e:
            logger.error(f"❌ RAG setup test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_document_insertion_and_query(self):
        """Test document insertion and querying"""
        logger.info("Testing Document Insertion and Query...")
        
        try:
            # Create RAG instance
            rag = create_local_rag_anything(
                working_dir=str(self.test_dir),
                model_preset="fast"
            )
            
            await rag.initialize()
            
            # Create test document
            test_doc = self.test_dir / "test_doc.txt"
            test_content = """
            Python is a high-level programming language.
            It was created by Guido van Rossum in 1991.
            Python is known for its simplicity and readability.
            It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
            """
            
            with open(test_doc, 'w') as f:
                f.write(test_content)
            
            # Insert document
            logger.info("Inserting test document...")
            result = await rag.insert_file(test_doc)
            assert result is not None
            logger.info("✅ Document inserted successfully")
            
            # Test query
            logger.info("Testing query...")
            response = await rag.query("Who created Python and when?")
            assert isinstance(response, str)
            assert len(response) > 0
            logger.info(f"✅ Query response: {response[:100]}...")
            
            # Cleanup
            rag.cleanup()
            
        except Exception as e:
            logger.error(f"❌ Document insertion/query test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_multimodal_query(self):
        """Test multimodal query functionality"""
        logger.info("Testing Multimodal Query...")
        
        try:
            # Create RAG instance
            rag = create_local_rag_anything(
                working_dir=str(self.test_dir),
                model_preset="fast"
            )
            
            await rag.initialize()
            
            # Create multimodal content
            multimodal_content = [
                {
                    "type": "table",
                    "table_data": "Name,Age\nAlice,25\nBob,30\nCharlie,35",
                    "table_caption": ["Sample user data"]
                }
            ]
            
            # Test multimodal query
            response = await rag.query(
                query="What is the average age in the table?",
                multimodal_content=multimodal_content
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            logger.info(f"✅ Multimodal query response: {response[:100]}...")
            
            # Cleanup
            rag.cleanup()
            
        except Exception as e:
            logger.error(f"❌ Multimodal query test failed: {e}")
            raise


async def run_manual_tests():
    """Run tests manually without pytest"""
    logger.info("🚀 Running Manual Tests for Local RAG Integration")
    
    # Test local models
    model_tests = TestLocalModels()
    
    try:
        logger.info("\n" + "="*50)
        await model_tests.test_local_embedding_wrapper()
        
        logger.info("\n" + "="*50)
        await model_tests.test_local_llm_wrapper()
        
        logger.info("\n" + "="*50)
        # Skip vision test for now as it requires large models
        # await model_tests.test_local_vision_wrapper()
        
    except Exception as e:
        logger.error(f"Model tests failed: {e}")
        return False
    
    # Test RAG integration
    rag_tests = TestRAGIntegration()
    
    try:
        logger.info("\n" + "="*50)
        rag_tests.setup_method()
        await rag_tests.test_basic_rag_setup()
        rag_tests.teardown_method()
        
        logger.info("\n" + "="*50)
        rag_tests.setup_method()
        await rag_tests.test_document_insertion_and_query()
        rag_tests.teardown_method()
        
        logger.info("\n" + "="*50)
        rag_tests.setup_method()
        await rag_tests.test_multimodal_query()
        rag_tests.teardown_method()
        
    except Exception as e:
        logger.error(f"RAG integration tests failed: {e}")
        return False
    
    logger.info("\n🎉 All tests passed successfully!")
    return True


def quick_model_check():
    """Quick check to see if models can be imported and basic functions work"""
    logger.info("🔍 Running Quick Model Check...")
    
    try:
        # Test imports
        from local_models import (
            LocalLLMManager,
            LocalVisionManager, 
            LocalEmbeddingManager
        )
        logger.info("✅ All model classes imported successfully")
        
        # Test basic functionality without loading heavy models
        embedding_func, dim = create_local_embedding_func("all-mpnet")
        logger.info(f"✅ Embedding function created with dimension: {dim}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick model check failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Local RAG Integration")
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick checks only (no model loading)"
    )
    parser.add_argument(
        "--full",
        action="store_true", 
        help="Run full integration tests"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        # Run quick checks
        success = quick_model_check()
        sys.exit(0 if success else 1)
    elif args.full:
        # Run full tests
        success = asyncio.run(run_manual_tests())
        sys.exit(0 if success else 1)
    else:
        print("Usage: python test_local_rag_integration.py [--quick|--full]")
        print("  --quick: Quick import and basic functionality checks")
        print("  --full:  Complete integration tests with model loading")
        sys.exit(1)