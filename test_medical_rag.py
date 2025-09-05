#!/usr/bin/env python3
"""
Test script for Medical Multimodal RAG System
Verifies basic functionality of all components
Compatible with uv virtual environments
"""

import os
import sys
import logging
import tempfile
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Set environment variables for better compatibility
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src to path (works with both uv and pip environments)
project_root = Path(__file__).parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    # Fallback for different project structures
    sys.path.insert(0, str(project_root))

# Import components
from medical_rag import MedicalMultimodalRAG
from models.medical_encoder import MedicalImageEncoder
from models.medical_retriever import MedicalKnowledgeRetriever
from models.medical_generator import MedicalAnswerGenerator
from data.medical_preprocessor import MedicalTextPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_medical_images(num_images: int = 5) -> list:
    """Create dummy medical images for testing"""
    images = []
    
    # Create different types of medical images
    image_types = [
        {"name": "chest_xray", "color": (200, 200, 200), "pattern": "cross"},
        {"name": "ct_scan", "color": (100, 100, 100), "pattern": "circle"},
        {"name": "mri", "color": (150, 150, 150), "pattern": "rectangle"},
        {"name": "histology", "color": (255, 200, 200), "pattern": "dots"},
        {"name": "ultrasound", "color": (50, 50, 50), "pattern": "lines"}
    ]
    
    for i in range(num_images):
        img_type = image_types[i % len(image_types)]
        
        # Create base image
        image = Image.new('RGB', (224, 224), color=img_type["color"])
        draw = ImageDraw.Draw(image)
        
        # Add pattern to distinguish images
        if img_type["pattern"] == "cross":
            draw.line([(112, 50), (112, 174)], fill=(255, 255, 255), width=3)
            draw.line([(50, 112), (174, 112)], fill=(255, 255, 255), width=3)
        elif img_type["pattern"] == "circle":
            draw.ellipse([50, 50, 174, 174], outline=(255, 255, 255), width=3)
        elif img_type["pattern"] == "rectangle":
            draw.rectangle([50, 50, 174, 174], outline=(255, 255, 255), width=3)
        elif img_type["pattern"] == "dots":
            for x in range(60, 180, 20):
                for y in range(60, 180, 20):
                    draw.ellipse([x-3, y-3, x+3, y+3], fill=(255, 255, 255))
        elif img_type["pattern"] == "lines":
            for y in range(60, 180, 10):
                draw.line([(60, y), (180, y)], fill=(255, 255, 255), width=1)
        
        images.append({
            'image': image,
            'type': img_type["name"],
            'metadata': {
                'modality': img_type["name"],
                'condition': ['normal', 'abnormal', 'pneumonia', 'fracture', 'tumor'][i % 5],
                'anatomy': ['chest', 'brain', 'abdomen', 'extremity', 'pelvis'][i % 5],
                'description': f"Medical {img_type['name']} image showing {['normal anatomy', 'pathological changes', 'inflammatory process', 'traumatic injury', 'mass lesion'][i % 5]}"
            }
        })
    
    return images


def create_dummy_medical_texts(num_texts: int = 5) -> list:
    """Create dummy medical texts for testing"""
    texts = [
        {
            'text': "Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli. Symptoms include cough, chest pain, fever, and difficulty breathing.",
            'metadata': {
                'source': 'medical_encyclopedia',
                'topic': 'respiratory_diseases',
                'specialty': 'pulmonology'
            }
        },
        {
            'text': "Chest X-rays are the most common imaging study performed in emergency departments. They are used to evaluate the lungs, heart, and chest wall for various pathological conditions.",
            'metadata': {
                'source': 'radiology_textbook',
                'topic': 'chest_imaging',
                'specialty': 'radiology'
            }
        },
        {
            'text': "Fractures are breaks in bone continuity. They can be classified as simple (closed) or compound (open), depending on whether the skin is broken.",
            'metadata': {
                'source': 'orthopedic_manual',
                'topic': 'bone_fractures',
                'specialty': 'orthopedics'
            }
        },
        {
            'text': "MRI uses strong magnetic fields and radio waves to generate detailed images of organs and tissues. It is particularly useful for soft tissue evaluation.",
            'metadata': {
                'source': 'imaging_handbook',
                'topic': 'mri_basics',
                'specialty': 'radiology'
            }
        },
        {
            'text': "Histopathology is the microscopic examination of tissue to study the manifestations of disease. It is essential for cancer diagnosis and treatment planning.",
            'metadata': {
                'source': 'pathology_textbook',
                'topic': 'tissue_analysis',
                'specialty': 'pathology'
            }
        }
    ]
    
    return texts[:num_texts]


def test_medical_preprocessor():
    """Test medical text preprocessor"""
    logger.info("Testing Medical Text Preprocessor...")
    
    try:
        preprocessor = MedicalTextPreprocessor()
        
        # Test abbreviation expansion
        test_text = "Pt w/ MI and CHF, s/p CABG. EKG shows AF."
        expanded = preprocessor.expand_abbreviations(test_text)
        logger.info(f"Abbreviation expansion: '{test_text}' -> '{expanded}'")
        
        # Test entity extraction
        medical_text = "Patient presents with chest pain, elevated troponins, and EKG changes consistent with acute MI."
        entities = preprocessor.extract_medical_entities(medical_text)
        logger.info(f"Medical entities: {entities}")
        
        # Test query preprocessing
        query = "Show me chest X-rays with pneumonia or lung infection"
        processed_query = preprocessor.preprocess_medical_query(query)
        logger.info(f"Query preprocessing: '{query}' -> '{processed_query}'")
        
        logger.info("✅ Medical Text Preprocessor test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Medical Text Preprocessor test failed: {e}")
        return False


def test_medical_encoder():
    """Test medical image encoder"""
    logger.info("Testing Medical Image Encoder...")
    
    try:
        encoder = MedicalImageEncoder()
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # Test image encoding
        image_embedding = encoder.encode_image(test_image, modality='chest_xray')
        logger.info(f"Image embedding shape: {image_embedding.shape}")
        
        # Test text encoding
        medical_query = "chest x-ray showing pneumonia"
        text_embedding = encoder.encode_text(medical_query)
        logger.info(f"Text embedding shape: {text_embedding.shape}")
        
        # Test similarity computation
        similarity = encoder.compute_similarity(
            image_embedding.reshape(1, -1),
            text_embedding.reshape(1, -1)
        )
        logger.info(f"Similarity score: {similarity[0]:.4f}")
        
        # Test batch encoding
        test_images = [test_image] * 3
        batch_embeddings = encoder.encode_images_batch(test_images)
        logger.info(f"Batch embeddings shape: {batch_embeddings.shape}")
        
        logger.info("✅ Medical Image Encoder test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Medical Image Encoder test failed: {e}")
        return False


def test_medical_retriever():
    """Test medical knowledge retriever"""
    logger.info("Testing Medical Knowledge Retriever...")
    
    try:
        retriever = MedicalKnowledgeRetriever(embedding_dim=512)
        
        # Create dummy embeddings
        image_embeddings = np.random.randn(10, 512).astype(np.float32)
        text_embeddings = np.random.randn(5, 512).astype(np.float32)
        
        # Create metadata
        image_metadata = [
            {
                'image_id': f"img_{i}",
                'modality': ['chest_xray', 'ct_scan', 'mri'][i % 3],
                'condition': ['normal', 'pneumonia', 'fracture'][i % 3],
                'anatomy': ['chest', 'brain', 'bone'][i % 3]
            }
            for i in range(10)
        ]
        
        text_metadata = [
            {
                'text_id': f"text_{i}",
                'source': 'medical_textbook',
                'topic': ['cardiology', 'radiology', 'pathology'][i % 3]
            }
            for i in range(5)
        ]
        
        # Add to retriever
        retriever.add_images(image_embeddings, image_metadata)
        retriever.add_texts(text_embeddings, text_metadata)
        
        # Test search
        query_embedding = np.random.randn(512).astype(np.float32)
        
        # Search images
        image_results = retriever.search_images(query_embedding, k=3)
        logger.info(f"Image search returned {len(image_results)} results")
        
        # Search texts
        text_results = retriever.search_texts(query_embedding, k=2)
        logger.info(f"Text search returned {len(text_results)} results")
        
        # Multimodal search
        multimodal_results = retriever.search_multimodal(query_embedding)
        logger.info(f"Multimodal search: {len(multimodal_results['combined'])} total results")
        
        # Test filtering
        filtered_results = retriever.search_images(
            query_embedding, k=5, filter_metadata={'modality': 'chest_xray'}
        )
        logger.info(f"Filtered search returned {len(filtered_results)} results")
        
        # Test statistics
        stats = retriever.get_statistics()
        logger.info(f"Retriever statistics: {stats}")
        
        logger.info("✅ Medical Knowledge Retriever test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Medical Knowledge Retriever test failed: {e}")
        return False


def test_medical_generator():
    """Test medical answer generator"""
    logger.info("Testing Medical Answer Generator...")
    
    try:
        generator = MedicalAnswerGenerator()
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='lightgray')
        
        # Test basic generation
        test_query = "What abnormalities are visible in this chest X-ray?"
        result = generator.generate_answer(test_image, test_query)
        
        logger.info(f"Generated answer: {result['answer']}")
        logger.info(f"Confidence: {result['confidence']:.3f}")
        
        # Test with retrieved context
        mock_context = [
            {
                'type': 'image',
                'score': 0.85,
                'metadata': {
                    'condition': 'pneumonia',
                    'findings': 'bilateral infiltrates'
                }
            }
        ]
        
        result_with_context = generator.generate_answer(
            test_image, 
            test_query,
            retrieved_context=mock_context
        )
        
        logger.info(f"Answer with context: {result_with_context['answer']}")
        logger.info(f"Context used: {result_with_context['retrieved_context_used']} items")
        
        logger.info("✅ Medical Answer Generator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Medical Answer Generator test failed: {e}")
        return False


def test_integrated_system():
    """Test the complete integrated Medical RAG system"""
    logger.info("Testing Integrated Medical RAG System...")
    
    try:
        # Initialize system
        medical_rag = MedicalMultimodalRAG()
        
        # Create dummy data
        image_data = []
        dummy_images = create_dummy_medical_images(3)
        
        for img_data in dummy_images:
            image_data.append({
                'image_path': img_data['image'],
                'metadata': img_data['metadata'],
                'modality': img_data['type']
            })
        
        text_data = create_dummy_medical_texts(3)
        
        # Build knowledge base
        logger.info("Building knowledge base...")
        medical_rag.build_knowledge_base(
            image_data=image_data,
            text_data=text_data,
            save_index=False
        )
        
        # Test multimodal query (with image)
        test_image = Image.new('RGB', (224, 224), color='white')
        test_question = "What abnormalities are visible in this medical image?"
        
        logger.info("Testing multimodal query...")
        result1 = medical_rag.query(
            image=test_image,
            question=test_question,
            k_retrieve=3
        )
        
        logger.info(f"Multimodal query result:")
        logger.info(f"  Answer: {result1['answer']}")
        logger.info(f"  Confidence: {result1.get('confidence', 'N/A')}")
        logger.info(f"  Response time: {result1['response_time']:.3f}s")
        
        # Test text-only query
        text_question = "What are the typical signs of pneumonia on chest imaging?"
        
        logger.info("Testing text-only query...")
        result2 = medical_rag.query(
            question=text_question,
            k_retrieve=2
        )
        
        logger.info(f"Text-only query result:")
        logger.info(f"  Answer: {result2['answer']}")
        logger.info(f"  Response time: {result2['response_time']:.3f}s")
        
        # Test batch queries
        batch_queries = [
            {'image': test_image, 'question': 'Describe this medical image'},
            {'question': 'What is pneumonia?'}
        ]
        
        logger.info("Testing batch queries...")
        batch_results = medical_rag.batch_query(batch_queries)
        logger.info(f"Batch processing completed: {len(batch_results)} results")
        
        # Get system statistics
        stats = medical_rag.get_system_stats()
        logger.info(f"System statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ Integrated Medical RAG System test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated Medical RAG System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Test save and load functionality"""
    logger.info("Testing Save/Load functionality...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and build system
            medical_rag = MedicalMultimodalRAG()
            
            # Create minimal knowledge base
            dummy_images = create_dummy_medical_images(2)
            image_data = []
            
            for img_data in dummy_images:
                image_data.append({
                    'image_path': img_data['image'],
                    'metadata': img_data['metadata'],
                    'modality': img_data['type']
                })
            
            medical_rag.build_knowledge_base(image_data=image_data, save_index=False)
            
            # Save system
            save_path = Path(temp_dir) / "test_system"
            medical_rag.save_system(str(save_path))
            logger.info(f"System saved to {save_path}")
            
            # Test loading knowledge base
            new_rag = MedicalMultimodalRAG()
            new_rag.load_knowledge_base(str(save_path / "knowledge_index"))
            logger.info("Knowledge base loaded successfully")
            
            # Test query on loaded system
            result = new_rag.query(question="Test query")
            logger.info(f"Query on loaded system successful: {result['response_time']:.3f}s")
            
        logger.info("✅ Save/Load functionality test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Save/Load functionality test failed: {e}")
        return False


def run_all_tests():
    """Run all test functions"""
    logger.info("=" * 60)
    logger.info("Starting Medical Multimodal RAG System Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # List of tests to run
    tests = [
        ("Medical Text Preprocessor", test_medical_preprocessor),
        ("Medical Image Encoder", test_medical_encoder),
        ("Medical Knowledge Retriever", test_medical_retriever),
        ("Medical Answer Generator", test_medical_generator),
        ("Integrated System", test_integrated_system),
        ("Save/Load Functionality", test_save_load)
    ]
    
    # Run each test
    for test_name, test_func in tests:
        logger.info("-" * 40)
        logger.info(f"Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        status_icon = "✅" if result else "❌"
        logger.info(f"{status_icon} {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("-" * 40)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Medical RAG system is ready.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")
    
    return passed == total


def main():
    """Main function for running tests"""
    return run_all_tests()

if __name__ == "__main__":
    # Run all tests
    success = main()
    
    if success:
        print("\n🚀 System is ready for use!")
        print("Next steps:")
        print("1. With uv (recommended):")
        print("   - uv add <new-dependencies>  # Add packages")
        print("   - uv run python your_script.py  # Run scripts")
        print("2. Or with pip (slower):")
        print("   - pip install -r requirements.txt")
        print("3. Download medical datasets (VQA-Med, PathVQA)")
        print("4. Build knowledge base with real medical data")
        print("5. Start developing the web demo interface")
        print("\n⚡ Tip: Use './scripts/dev-workflow.sh help' for development commands")
    else:
        print("\n🔧 Some tests failed. Please check the implementation.")
        print("💡 Try running with uv: 'uv run python test_medical_rag.py'")
    
    sys.exit(0 if success else 1)