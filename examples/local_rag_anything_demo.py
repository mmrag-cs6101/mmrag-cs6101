"""
Local RAG-Anything Demo
Demonstrates how to use RAG-Anything with local open-source models
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import our local RAG-Anything integration
from local_models import create_local_rag_anything, MedicalRAGAnything

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalRAGDemo:
    """Demo class for Local RAG-Anything"""
    
    def __init__(self, working_dir: str = "./demo_rag_storage"):
        self.working_dir = working_dir
        self.rag = None
    
    async def setup_rag_system(self, model_preset: str = "fast"):
        """Setup the RAG system with local models"""
        logger.info(f"Setting up Local RAG-Anything with preset: {model_preset}")
        
        try:
            # Create Local RAG-Anything instance
            self.rag = create_local_rag_anything(
                working_dir=self.working_dir,
                model_preset=model_preset
            )
            
            # Initialize the system
            await self.rag.initialize()
            
            logger.info("✅ RAG system setup complete!")
            
            # Show model info
            info = self.rag.get_model_info()
            logger.info(f"Loaded models: {info}")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG system: {e}")
            raise
    
    async def demo_text_processing(self):
        """Demo text document processing"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Text Document Processing")
        logger.info("="*50)
        
        try:
            # Create a sample text document
            sample_doc = Path(self.working_dir) / "sample_document.txt"
            sample_doc.parent.mkdir(parents=True, exist_ok=True)
            
            sample_content = """
            Artificial Intelligence and Machine Learning
            
            Artificial Intelligence (AI) is a field of computer science that aims to create 
            intelligent machines that work and react like humans. Machine Learning (ML) is 
            a subset of AI that provides systems the ability to automatically learn and 
            improve from experience without being explicitly programmed.
            
            Deep Learning is a subset of machine learning that uses neural networks with 
            multiple layers to model and understand complex patterns in data. It has been 
            particularly successful in areas like image recognition, natural language 
            processing, and speech recognition.
            
            Applications of AI include:
            - Autonomous vehicles
            - Medical diagnosis
            - Financial trading
            - Recommendation systems
            - Virtual assistants
            """
            
            with open(sample_doc, 'w') as f:
                f.write(sample_content)
            
            logger.info(f"Created sample document: {sample_doc}")
            
            # Insert the document
            logger.info("Inserting document into RAG system...")
            result = await self.rag.insert_file(sample_doc)
            logger.info(f"✅ Document inserted: {result}")
            
            # Test queries
            queries = [
                "What is artificial intelligence?",
                "How is machine learning different from deep learning?",
                "What are some applications of AI?",
                "Explain the relationship between AI, ML, and deep learning."
            ]
            
            for query in queries:
                logger.info(f"\nQuery: {query}")
                response = await self.rag.query(query)
                logger.info(f"Response: {response}")
                logger.info("-" * 40)
        
        except Exception as e:
            logger.error(f"Text processing demo failed: {e}")
    
    async def demo_multimodal_processing(self):
        """Demo multimodal content processing"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Multimodal Content Processing")
        logger.info("="*50)
        
        try:
            # Create sample multimodal content
            multimodal_content = [
                {
                    "type": "table",
                    "table_data": """
                    | Model Type | Parameters | Use Case |
                    |------------|------------|----------|
                    | GPT-4      | 1.7T       | General AI |
                    | LLaMA      | 70B        | Open Source LLM |
                    | CLIP       | 400M       | Vision-Language |
                    | DALL-E     | 12B        | Image Generation |
                    """,
                    "table_caption": ["Comparison of different AI models"]
                },
                {
                    "type": "equation", 
                    "text": "Loss = -∑(y * log(ŷ))",
                    "text_format": "LaTeX"
                }
            ]
            
            # Test multimodal query
            query = "Compare the different AI models in the table and explain the loss function."
            
            logger.info(f"Multimodal Query: {query}")
            response = await self.rag.query(
                query=query,
                multimodal_content=multimodal_content
            )
            logger.info(f"Response: {response}")
        
        except Exception as e:
            logger.error(f"Multimodal processing demo failed: {e}")
    
    async def demo_medical_rag(self):
        """Demo medical-specialized RAG"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Medical RAG System")
        logger.info("="*50)
        
        try:
            # Create medical RAG instance
            medical_rag = MedicalRAGAnything(
                working_dir="./demo_medical_rag_storage"
            )
            
            await medical_rag.initialize()
            
            # Create sample medical document
            medical_doc = Path("./demo_medical_rag_storage") / "medical_report.txt"
            medical_doc.parent.mkdir(parents=True, exist_ok=True)
            
            medical_content = """
            Patient Medical Report
            
            Patient ID: 12345
            Date: 2024-01-15
            
            Chief Complaint: Chest pain and shortness of breath
            
            History of Present Illness:
            The patient is a 65-year-old male with a history of HTN and DM who presents 
            with acute onset chest pain. The pain began approximately 2 hours ago and is 
            described as crushing, substernal, radiating to the left arm. Patient also 
            reports SOB and diaphoresis.
            
            Physical Examination:
            - BP: 180/100 mmHg
            - HR: 110 bpm
            - Temperature: 98.6°F
            - Cardiovascular: S1, S2 present, no murmurs
            - Pulmonary: Clear to auscultation bilaterally
            
            Laboratory Results:
            - Troponin I: 0.8 ng/mL (elevated)
            - CBC: WBC 8.5, Hgb 14.2, Plt 250
            - BNP: 450 pg/mL
            
            Imaging:
            - ECG: ST elevation in leads II, III, aVF
            - Chest X-ray: No acute cardiopulmonary process
            
            Assessment and Plan:
            1. STEMI (ST-elevation myocardial infarction) - likely RCA occlusion
            2. HTN - continue current medications
            3. DM - monitor glucose closely
            
            Plan: Emergency cardiac catheterization, dual antiplatelet therapy, 
            statin therapy, beta-blocker when stable.
            """
            
            with open(medical_doc, 'w') as f:
                f.write(medical_content)
            
            # Insert medical document
            logger.info("Inserting medical document...")
            result = await medical_rag.insert_medical_document(
                medical_doc,
                document_type="clinical_notes",
                patient_id="12345"
            )
            logger.info(f"✅ Medical document inserted")
            
            # Test medical queries
            medical_queries = [
                "What is the patient's primary diagnosis?",
                "What are the key lab findings?", 
                "What does STEMI stand for and what causes it?",
                "What medications should be started for this patient?",
                "What do the ECG findings suggest?"
            ]
            
            for query in medical_queries:
                logger.info(f"\nMedical Query: {query}")
                response = await medical_rag.query(query)
                logger.info(f"Response: {response}")
                logger.info("-" * 40)
            
            # Cleanup medical RAG
            medical_rag.cleanup()
        
        except Exception as e:
            logger.error(f"Medical RAG demo failed: {e}")
    
    async def demo_batch_processing(self):
        """Demo batch document processing"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Batch Document Processing")
        logger.info("="*50)
        
        try:
            # Create a directory with multiple documents
            batch_dir = Path(self.working_dir) / "batch_docs"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Create multiple sample documents
            docs = {
                "ai_basics.txt": """
                Artificial Intelligence Basics
                AI is the simulation of human intelligence in machines.
                It includes machine learning, deep learning, and neural networks.
                """,
                "ml_algorithms.txt": """
                Machine Learning Algorithms
                Common algorithms include linear regression, decision trees,
                random forests, and support vector machines.
                """,
                "deep_learning.txt": """
                Deep Learning Overview
                Deep learning uses neural networks with multiple layers.
                Popular frameworks include TensorFlow, PyTorch, and Keras.
                """
            }
            
            for filename, content in docs.items():
                doc_path = batch_dir / filename
                with open(doc_path, 'w') as f:
                    f.write(content)
            
            logger.info(f"Created {len(docs)} documents in {batch_dir}")
            
            # Batch process the directory
            logger.info("Batch processing directory...")
            result = await self.rag.insert_directory(batch_dir)
            logger.info(f"✅ Batch processing complete: {result}")
            
            # Test cross-document queries
            cross_queries = [
                "Compare machine learning and deep learning",
                "What are the different types of AI mentioned across all documents?",
                "Which frameworks are mentioned for deep learning?"
            ]
            
            for query in cross_queries:
                logger.info(f"\nCross-document Query: {query}")
                response = await self.rag.query(query)
                logger.info(f"Response: {response}")
                logger.info("-" * 40)
        
        except Exception as e:
            logger.error(f"Batch processing demo failed: {e}")
    
    async def run_full_demo(self, model_preset: str = "fast"):
        """Run the complete demo"""
        logger.info("🚀 Starting Local RAG-Anything Demo")
        logger.info(f"Model preset: {model_preset}")
        
        try:
            # Setup RAG system
            await self.setup_rag_system(model_preset)
            
            # Run individual demos
            await self.demo_text_processing()
            await self.demo_multimodal_processing() 
            await self.demo_batch_processing()
            await self.demo_medical_rag()
            
            logger.info("\n🎉 Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
        finally:
            # Cleanup
            if self.rag:
                self.rag.cleanup()
    
    def cleanup(self):
        """Clean up demo files"""
        import shutil
        
        for dir_path in [self.working_dir, "./demo_medical_rag_storage"]:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)
                logger.info(f"Cleaned up: {dir_path}")


async def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local RAG-Anything Demo")
    parser.add_argument(
        "--preset", 
        choices=["fast", "balanced", "quality", "medical"],
        default="fast",
        help="Model preset to use"
    )
    parser.add_argument(
        "--cleanup", 
        action="store_true",
        help="Clean up demo files after completion"
    )
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = LocalRAGDemo()
    
    try:
        await demo.run_full_demo(args.preset)
    finally:
        if args.cleanup:
            demo.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())