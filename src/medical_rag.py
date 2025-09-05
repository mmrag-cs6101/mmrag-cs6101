"""
Main Medical Multimodal RAG System
Integrates all components: encoder, retriever, generator, and preprocessor
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from PIL import Image
import torch

# Import all components
from src.models.medical_encoder import MedicalImageEncoder
from src.models.medical_retriever import MedicalKnowledgeRetriever
from src.models.medical_generator import MedicalAnswerGenerator
from src.data.medical_preprocessor import MedicalTextPreprocessor

logger = logging.getLogger(__name__)


class MedicalMultimodalRAG:
    """
    Complete Medical Multimodal RAG System
    """
    
    def __init__(
        self,
        encoder_model: str = "openai/clip-vit-base-patch32",
        generator_model: str = "Salesforce/blip-image-captioning-large",
        embedding_dim: int = 512,
        device: Optional[str] = None,
        cache_dir: Optional[str] = "./cache",
        index_type: str = "flat",
        use_gpu_index: bool = False
    ):
        """
        Initialize Medical Multimodal RAG System
        
        Args:
            encoder_model: CLIP model for image/text encoding
            generator_model: BLIP model for answer generation
            embedding_dim: Embedding dimension
            device: Device to run models on
            cache_dir: Directory to cache models and data
            index_type: FAISS index type
            use_gpu_index: Whether to use GPU for FAISS index
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing Medical RAG components...")
        
        # Text preprocessor (no GPU needed)
        self.preprocessor = MedicalTextPreprocessor()
        
        # Image encoder
        logger.info("Loading medical image encoder...")
        self.encoder = MedicalImageEncoder(
            model_name=encoder_model,
            device=self.device,
            cache_dir=str(self.cache_dir / "encoder")
        )
        
        # Knowledge retriever
        logger.info("Initializing knowledge retriever...")
        self.retriever = MedicalKnowledgeRetriever(
            embedding_dim=embedding_dim,
            index_type=index_type,
            metric="cosine"
        )
        
        # Answer generator
        logger.info("Loading medical answer generator...")
        self.generator = MedicalAnswerGenerator(
            model_name=generator_model,
            device=self.device,
            cache_dir=str(self.cache_dir / "generator")
        )
        
        # System state
        self.is_knowledge_base_built = False
        self.stats = {
            'total_queries': 0,
            'total_images_indexed': 0,
            'total_texts_indexed': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"Medical RAG system initialized on {self.device}")
    
    def build_knowledge_base(
        self,
        image_data: List[Dict[str, Any]],
        text_data: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 16,
        save_index: bool = True
    ):
        """
        Build medical knowledge base from images and texts
        
        Args:
            image_data: List of dicts with 'image_path', 'metadata', 'modality'
            text_data: Optional list of dicts with 'text', 'metadata'
            batch_size: Batch size for processing
            save_index: Whether to save the built index
        """
        logger.info(f"Building knowledge base with {len(image_data)} images")
        
        # Process images in batches
        image_embeddings_list = []
        image_metadata_list = []
        
        for i in range(0, len(image_data), batch_size):
            batch_data = image_data[i:i + batch_size]
            
            # Prepare batch
            batch_images = []
            batch_modalities = []
            batch_metadata = []
            
            for item in batch_data:
                try:
                    # Load image
                    image_path = item['image_path']
                    if isinstance(image_path, str):
                        image = Image.open(image_path)
                    else:
                        image = image_path  # Already PIL Image
                    
                    batch_images.append(image)
                    batch_modalities.append(item.get('modality', 'general'))
                    
                    # Enhance metadata with preprocessed text
                    metadata = item['metadata'].copy()
                    if 'description' in metadata:
                        processed_desc = self.preprocessor.preprocess_medical_query(
                            metadata['description']
                        )
                        metadata['processed_description'] = processed_desc
                        metadata['medical_keywords'] = self.preprocessor.create_medical_keywords(
                            metadata['description']
                        )
                    
                    batch_metadata.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Error processing image {item.get('image_path', 'unknown')}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Encode batch
            try:
                batch_embeddings = self.encoder.encode_images_batch(
                    batch_images, 
                    batch_modalities, 
                    batch_size=len(batch_images)
                )
                image_embeddings_list.append(batch_embeddings)
                image_metadata_list.extend(batch_metadata)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(image_data)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
                continue
        
        # Combine all embeddings
        if image_embeddings_list:
            all_image_embeddings = np.vstack(image_embeddings_list)
            
            # Add to retriever
            self.retriever.add_images(all_image_embeddings, image_metadata_list)
            self.stats['total_images_indexed'] = len(image_metadata_list)
            
            logger.info(f"Added {len(image_metadata_list)} images to knowledge base")
        
        # Process text data if provided
        if text_data:
            logger.info(f"Processing {len(text_data)} text documents")
            
            text_embeddings_list = []
            text_metadata_list = []
            
            for i in range(0, len(text_data), batch_size):
                batch_texts = text_data[i:i + batch_size]
                
                # Extract texts and process metadata
                texts = []
                metadata = []
                
                for item in batch_texts:
                    text = item['text']
                    processed_text = self.preprocessor.preprocess_medical_document(text)
                    
                    texts.append(processed_text['enhanced_text'])
                    
                    # Enhanced metadata
                    meta = item['metadata'].copy()
                    meta.update({
                        'original_text': text,
                        'processed_text': processed_text['processed_text'],
                        'entities': processed_text['entities'],
                        'medical_keywords': self.preprocessor.create_medical_keywords(text)
                    })
                    metadata.append(meta)
                
                # Encode texts
                try:
                    batch_embeddings = self.encoder.encode_text(texts)
                    if len(batch_embeddings.shape) == 1:
                        batch_embeddings = batch_embeddings.reshape(1, -1)
                    
                    text_embeddings_list.append(batch_embeddings)
                    text_metadata_list.extend(metadata)
                    
                except Exception as e:
                    logger.error(f"Error encoding text batch: {e}")
                    continue
            
            # Combine text embeddings
            if text_embeddings_list:
                all_text_embeddings = np.vstack(text_embeddings_list)
                self.retriever.add_texts(all_text_embeddings, text_metadata_list)
                self.stats['total_texts_indexed'] = len(text_metadata_list)
                
                logger.info(f"Added {len(text_metadata_list)} texts to knowledge base")
        
        # Save index if requested
        if save_index:
            index_path = self.cache_dir / "knowledge_index"
            self.retriever.save_index(str(index_path))
            logger.info(f"Knowledge base saved to {index_path}")
        
        self.is_knowledge_base_built = True
        logger.info("Knowledge base building completed")
    
    def load_knowledge_base(self, index_path: str):
        """
        Load pre-built knowledge base
        
        Args:
            index_path: Path to saved index
        """
        try:
            self.retriever = MedicalKnowledgeRetriever.load_index(index_path)
            self.is_knowledge_base_built = True
            
            stats = self.retriever.get_statistics()
            self.stats['total_images_indexed'] = stats.get('total_images', 0)
            self.stats['total_texts_indexed'] = stats.get('total_texts', 0)
            
            logger.info(f"Knowledge base loaded from {index_path}")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise
    
    def query(
        self,
        image: Optional[Union[str, Image.Image]] = None,
        question: str = "",
        k_retrieve: int = 5,
        k_images: int = 3,
        k_texts: int = 2,
        include_confidence: bool = True,
        return_retrieved: bool = True
    ) -> Dict[str, Any]:
        """
        Main query interface for the Medical RAG system
        
        Args:
            image: Query image (optional for text-only queries)
            question: Medical question
            k_retrieve: Total number of items to retrieve
            k_images: Number of images to retrieve
            k_texts: Number of texts to retrieve
            include_confidence: Whether to include confidence scores
            return_retrieved: Whether to return retrieved items
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_knowledge_base_built:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess question
            processed_question = self.preprocessor.preprocess_medical_query(question)
            
            # Create query embedding
            if image is not None:
                # Multimodal query: combine image and text
                if isinstance(image, str):
                    image = Image.open(image)
                
                # Get image embedding
                image_embedding = self.encoder.encode_image(image)
                
                # Get text embedding
                text_embedding = self.encoder.encode_text(processed_question)
                
                # Combine embeddings (simple average - can be improved)
                query_embedding = (image_embedding + text_embedding) / 2
                
            else:
                # Text-only query
                query_embedding = self.encoder.encode_text(processed_question)
            
            # Retrieve relevant information
            retrieved_results = self.retriever.search_multimodal(
                query_embedding,
                k_images=k_images,
                k_texts=k_texts,
                combine_scores=True
            )
            
            # Generate answer
            if image is not None:
                answer_result = self.generator.generate_answer(
                    image=image,
                    query=question,
                    retrieved_context=retrieved_results['combined'][:k_retrieve]
                )
            else:
                # For text-only queries, use most relevant image if available
                if retrieved_results['images']:
                    top_image_metadata = retrieved_results['images'][0]['metadata']
                    if 'file_path' in top_image_metadata:
                        reference_image = Image.open(top_image_metadata['file_path'])
                        answer_result = self.generator.generate_answer(
                            image=reference_image,
                            query=question,
                            retrieved_context=retrieved_results['combined'][:k_retrieve]
                        )
                    else:
                        # No image available - generate text-based answer
                        answer_result = {
                            'answer': f"Based on the medical knowledge: {processed_question}. " + 
                                     "However, image analysis would be needed for a complete assessment.",
                            'confidence': 0.6,
                            'raw_answer': "",
                            'query': question
                        }
                else:
                    answer_result = {
                        'answer': "I need an image to provide a comprehensive medical analysis.",
                        'confidence': 0.3,
                        'raw_answer': "",
                        'query': question
                    }
            
            # Prepare response
            response_time = time.time() - start_time
            
            result = {
                'answer': answer_result['answer'],
                'query': question,
                'processed_query': processed_question,
                'response_time': response_time,
                'has_image': image is not None
            }
            
            if include_confidence:
                result['confidence'] = answer_result.get('confidence', 0.5)
            
            if return_retrieved:
                result['retrieved_items'] = retrieved_results['combined'][:k_retrieve]
                result['retrieval_stats'] = {
                    'total_retrieved': len(retrieved_results['combined']),
                    'images_retrieved': len(retrieved_results['images']),
                    'texts_retrieved': len(retrieved_results['texts'])
                }
            
            # Update stats
            self.stats['total_queries'] += 1
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + response_time) /
                self.stats['total_queries']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question.",
                'error': str(e),
                'query': question,
                'response_time': time.time() - start_time
            }
    
    def batch_query(
        self,
        queries: List[Dict[str, Any]],
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batches
        
        Args:
            queries: List of query dictionaries with 'image' and 'question' keys
            batch_size: Batch size for processing
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            batch_results = []
            for query_dict in batch_queries:
                result = self.query(
                    image=query_dict.get('image'),
                    question=query_dict.get('question', ''),
                    k_retrieve=query_dict.get('k_retrieve', 5)
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(queries)-1)//batch_size + 1}")
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = self.stats.copy()
        stats['knowledge_base_built'] = self.is_knowledge_base_built
        stats['device'] = self.device
        stats['models'] = {
            'encoder': self.encoder.model_name,
            'generator': self.generator.model_name
        }
        
        if self.is_knowledge_base_built:
            retriever_stats = self.retriever.get_statistics()
            stats.update(retriever_stats)
        
        return stats
    
    def save_system(self, save_dir: str):
        """
        Save the entire system
        
        Args:
            save_dir: Directory to save system components
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save knowledge base
        if self.is_knowledge_base_built:
            self.retriever.save_index(str(save_path / "knowledge_index"))
        
        # Save models (if fine-tuned)
        self.encoder.save_model(str(save_path / "encoder"))
        self.generator.save_model(str(save_path / "generator"))
        
        # Save system configuration and stats
        import json
        config = {
            'stats': self.stats,
            'device': self.device,
            'is_knowledge_base_built': self.is_knowledge_base_built
        }
        
        with open(save_path / "system_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Medical RAG system saved to {save_dir}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize system
    medical_rag = MedicalMultimodalRAG()
    
    # Create dummy knowledge base for testing
    dummy_image_data = [
        {
            'image_path': Image.new('RGB', (224, 224), color='white'),  # Dummy image
            'metadata': {
                'condition': 'pneumonia',
                'anatomy': 'chest',
                'modality': 'chest_xray',
                'description': 'Chest X-ray showing bilateral infiltrates consistent with pneumonia'
            },
            'modality': 'chest_xray'
        }
    ]
    
    dummy_text_data = [
        {
            'text': 'Pneumonia is an infection that causes inflammation in one or both lungs. On chest X-rays, it typically appears as areas of increased opacity or consolidation.',
            'metadata': {
                'source': 'medical_textbook',
                'topic': 'respiratory_diseases'
            }
        }
    ]
    
    # Build knowledge base
    print("Building knowledge base...")
    medical_rag.build_knowledge_base(
        image_data=dummy_image_data,
        text_data=dummy_text_data,
        save_index=False  # Don't save for testing
    )
    
    # Test queries
    test_queries = [
        {
            'image': Image.new('RGB', (224, 224), color='gray'),
            'question': 'What abnormalities are visible in this chest X-ray?'
        },
        {
            'question': 'What are the typical signs of pneumonia on chest imaging?'
        }
    ]
    
    # Process queries
    for i, query in enumerate(test_queries):
        print(f"\n--- Query {i+1} ---")
        print(f"Question: {query['question']}")
        
        result = medical_rag.query(**query)
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Response time: {result['response_time']:.3f}s")
        if 'retrieval_stats' in result:
            print(f"Retrieved items: {result['retrieval_stats']}")
    
    # Print system stats
    print(f"\n--- System Statistics ---")
    stats = medical_rag.get_system_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nMedical RAG system test completed!")