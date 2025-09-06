"""
Quick test to verify the API fix
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

async def test_api_fix():
    """Test the corrected API calls"""
    try:
        # Create RAG instance
        rag = create_local_rag_anything(
            working_dir="./test_api_fix_storage",
            model_preset="fast"
        )
        
        # Initialize
        logger.info("Initializing RAG system...")
        await rag.initialize()
        logger.info("✅ RAG system initialized")
        
        # Create test document
        test_dir = Path("./test_api_fix_storage")
        test_dir.mkdir(exist_ok=True)
        test_doc = test_dir / "test.txt"
        
        with open(test_doc, 'w') as f:
            f.write("This is a test document. It contains information about artificial intelligence and machine learning.")
        
        # Test document insertion
        logger.info("Testing document insertion...")
        result = await rag.insert_file(test_doc)
        logger.info(f"✅ Document inserted: {type(result)}")
        
        # Test query
        logger.info("Testing query...")
        response = await rag.query("What does this document discuss?")
        logger.info(f"✅ Query successful: {response[:100]}...")
        
        # Cleanup
        rag.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_api_fix())
    if success:
        print("✅ API fix successful!")
    else:
        print("❌ API fix failed!")
        sys.exit(1)