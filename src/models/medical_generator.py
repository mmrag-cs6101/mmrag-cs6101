"""
Medical Answer Generator using BLIP and other multimodal models
Generates medical answers based on retrieved images and context
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering, AutoProcessor, AutoModelForVision2Seq
)
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
import re

logger = logging.getLogger(__name__)


class MedicalAnswerGenerator:
    """
    Medical answer generator with multimodal understanding
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        model_type: str = "captioning",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_length: int = 150,
        num_beams: int = 4
    ):
        """
        Initialize medical answer generator
        
        Args:
            model_name: HuggingFace model name
            model_type: Type of model ('captioning', 'vqa', 'conditional')
            device: Device to run model on
            cache_dir: Directory to cache models
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_type = model_type
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Load model based on type
        self._load_model(model_name, model_type, cache_dir)
        
        # Medical context templates
        self.medical_templates = self._setup_medical_templates()
        
        # Medical answer post-processing rules
        self.medical_rules = self._setup_medical_rules()
        
        logger.info(f"Medical generator initialized: {model_name} on {self.device}")
    
    def _load_model(self, model_name: str, model_type: str, cache_dir: Optional[str]):
        """Load the appropriate model and processor"""
        try:
            if model_type == "captioning":
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self.processor = BlipProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
            elif model_type == "vqa":
                self.model = BlipForQuestionAnswering.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self.processor = BlipProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
            else:
                # Try to load as generic vision-to-seq model
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            # Fallback to BLIP captioning
            logger.info("Falling back to BLIP captioning model")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model.to(self.device)
            self.model.eval()
    
    def _setup_medical_templates(self) -> Dict[str, str]:
        """Setup medical-specific prompt templates"""
        return {
            'diagnosis': "Based on this medical image, what is the most likely diagnosis? ",
            'findings': "Describe the key medical findings visible in this image: ",
            'anatomy': "Identify the anatomical structures visible in this medical image: ",
            'abnormalities': "What abnormalities or pathological changes are visible in this image? ",
            'impression': "Provide a clinical impression of this medical image: ",
            'differential': "What are the differential diagnoses for the findings in this image? ",
            'normal': "Is this medical image normal or abnormal? Explain: ",
            'description': "Provide a detailed medical description of this image: "
        }
    
    def _setup_medical_rules(self) -> Dict[str, Any]:
        """Setup medical answer processing rules"""
        return {
            'uncertainty_phrases': [
                "most likely", "possibly", "may indicate", "could be",
                "consistent with", "suspicious for", "appears to be"
            ],
            'confidence_modifiers': {
                'high': ["clearly shows", "definitely", "obviously"],
                'medium': ["likely", "probably", "appears to"],
                'low': ["possibly", "may", "could", "might"]
            },
            'medical_disclaimers': [
                "Clinical correlation is recommended.",
                "Further evaluation may be needed.",
                "Consult with a medical professional for definitive diagnosis."
            ],
            'avoid_phrases': [
                "I am not a doctor", "This is not medical advice",
                "Please see a doctor"  # These are implicit
            ]
        }
    
    def preprocess_medical_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Preprocess medical query with context
        
        Args:
            query: Original medical query
            context: Additional medical context
            
        Returns:
            Enhanced query with medical context
        """
        # Detect query type
        query_lower = query.lower()
        query_type = "description"  # default
        
        if any(word in query_lower for word in ["diagnosis", "diagnose", "condition"]):
            query_type = "diagnosis"
        elif any(word in query_lower for word in ["findings", "find", "show"]):
            query_type = "findings"
        elif any(word in query_lower for word in ["anatomy", "structure", "organ"]):
            query_type = "anatomy"
        elif any(word in query_lower for word in ["abnormal", "pathology", "disease"]):
            query_type = "abnormalities"
        elif any(word in query_lower for word in ["normal", "healthy"]):
            query_type = "normal"
        
        # Use template if available
        if query_type in self.medical_templates:
            template = self.medical_templates[query_type]
            enhanced_query = template + query
        else:
            enhanced_query = query
        
        # Add context if provided
        if context:
            enhanced_query = f"Context: {context}. Question: {enhanced_query}"
        
        return enhanced_query
    
    def generate_answer(
        self,
        image: Union[str, Image.Image],
        query: str,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Generate medical answer for image and query
        
        Args:
            image: Medical image (path or PIL Image)
            query: Medical question
            retrieved_context: Retrieved similar cases/texts
            temperature: Generation temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated answer with metadata
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image)
            
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Build context from retrieved information
            context_text = self._build_context_from_retrieval(retrieved_context)
            
            # Preprocess query with context
            enhanced_query = self.preprocess_medical_query(query, context_text)
            
            # Prepare inputs based on model type
            if self.model_type == "vqa":
                inputs = self.processor(image, enhanced_query, return_tensors="pt")
            else:
                inputs = self.processor(image, enhanced_query, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generation parameters (avoid pad_token_id conflicts)
            generation_kwargs = {
                'max_length': self.max_length,
                'num_beams': self.num_beams,
                'early_stopping': True,
                'temperature': temperature,
                'do_sample': do_sample,
            }
            
            # Don't add pad_token_id - let the model handle it automatically
            
            # Generate answer
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode answer
            raw_answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Post-process medical answer
            processed_answer = self.post_process_medical_answer(
                raw_answer, query, retrieved_context
            )
            
            # Compute confidence score
            confidence_score = self._compute_confidence_score(
                processed_answer, retrieved_context
            )
            
            return {
                'answer': processed_answer,
                'raw_answer': raw_answer,
                'confidence': confidence_score,
                'query': query,
                'enhanced_query': enhanced_query,
                'retrieved_context_used': len(retrieved_context) if retrieved_context else 0,
                'generation_params': generation_kwargs
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': "I apologize, but I encountered an error while analyzing this medical image. Please try again or consult with a medical professional.",
                'raw_answer': "",
                'confidence': 0.0,
                'query': query,
                'enhanced_query': query,
                'retrieved_context_used': len(retrieved_context) if retrieved_context else 0,
                'error': str(e)
            }
    
    def _build_context_from_retrieval(
        self, 
        retrieved_context: Optional[List[Dict[str, Any]]]
    ) -> str:
        """
        Build context text from retrieved similar cases
        
        Args:
            retrieved_context: List of retrieved items with metadata
            
        Returns:
            Formatted context string
        """
        if not retrieved_context:
            return ""
        
        context_parts = []
        
        for item in retrieved_context[:3]:  # Use top 3 most relevant
            if item['type'] == 'image':
                metadata = item['metadata']
                if 'condition' in metadata:
                    context_parts.append(f"Similar case shows {metadata['condition']}")
                if 'findings' in metadata:
                    context_parts.append(f"Related findings: {metadata['findings']}")
            elif item['type'] == 'text':
                metadata = item['metadata']
                if 'text' in metadata:
                    # Truncate long texts
                    text = metadata['text'][:200] + "..." if len(metadata['text']) > 200 else metadata['text']
                    context_parts.append(f"Relevant information: {text}")
        
        return ". ".join(context_parts) if context_parts else ""
    
    def post_process_medical_answer(
        self,
        raw_answer: str,
        original_query: str,
        retrieved_context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Post-process generated answer for medical appropriateness
        
        Args:
            raw_answer: Raw generated answer
            original_query: Original query
            retrieved_context: Retrieved context items
            
        Returns:
            Post-processed medical answer
        """
        # Remove model artifacts
        processed_answer = raw_answer.strip()
        
        # Remove repetitive text (common in BLIP outputs)
        sentences = processed_answer.split('. ')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        processed_answer = '. '.join(unique_sentences)
        
        # Add medical uncertainty where appropriate
        if not any(phrase in processed_answer.lower() for phrase in self.medical_rules['uncertainty_phrases']):
            # Add appropriate medical uncertainty
            if "diagnosis" in original_query.lower():
                processed_answer = f"The imaging findings are {processed_answer}"
        
        # Ensure proper medical language
        processed_answer = self._enhance_medical_language(processed_answer)
        
        # Add clinical disclaimer if making diagnostic statements
        if any(word in processed_answer.lower() for word in ["diagnosis", "disease", "condition", "pathology"]):
            if not processed_answer.endswith('.'):
                processed_answer += '.'
            processed_answer += " Clinical correlation is recommended."
        
        return processed_answer
    
    def _enhance_medical_language(self, text: str) -> str:
        """
        Enhance text with appropriate medical terminology
        
        Args:
            text: Input text
            
        Returns:
            Enhanced text with medical language
        """
        # Replace common terms with medical equivalents
        replacements = {
            r'\blung\b': 'pulmonary',
            r'\bheart\b': 'cardiac',
            r'\bbrain\b': 'cerebral',
            r'\bkidney\b': 'renal',
            r'\bliver\b': 'hepatic',
            r'\bstomach\b': 'gastric'
        }
        
        enhanced_text = text
        for pattern, replacement in replacements.items():
            # Only replace if not already medical term
            if replacement not in enhanced_text.lower():
                enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
        
        return enhanced_text
    
    def _compute_confidence_score(
        self,
        answer: str,
        retrieved_context: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        Compute confidence score for generated answer
        
        Args:
            answer: Generated answer
            retrieved_context: Retrieved context items
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Adjust based on uncertainty language
        answer_lower = answer.lower()
        
        # High confidence indicators
        high_conf_phrases = self.medical_rules['confidence_modifiers']['high']
        if any(phrase in answer_lower for phrase in high_conf_phrases):
            confidence += 0.2
        
        # Medium confidence indicators
        medium_conf_phrases = self.medical_rules['confidence_modifiers']['medium']
        if any(phrase in answer_lower for phrase in medium_conf_phrases):
            confidence += 0.1
        
        # Low confidence indicators
        low_conf_phrases = self.medical_rules['confidence_modifiers']['low']
        if any(phrase in answer_lower for phrase in low_conf_phrases):
            confidence -= 0.1
        
        # Adjust based on retrieved context quality
        if retrieved_context:
            avg_retrieval_score = np.mean([item['score'] for item in retrieved_context])
            confidence += (avg_retrieval_score - 0.5) * 0.2  # Scale retrieval scores
        
        # Adjust based on answer length and detail
        if len(answer.split()) < 10:  # Very short answers
            confidence -= 0.1
        elif len(answer.split()) > 30:  # Detailed answers
            confidence += 0.1
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))
    
    def generate_batch_answers(
        self,
        images: List[Union[str, Image.Image]],
        queries: List[str],
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for batch of images and queries
        
        Args:
            images: List of images
            queries: List of queries
            batch_size: Batch size for processing
            
        Returns:
            List of generated answers
        """
        if len(images) != len(queries):
            raise ValueError("Number of images must match number of queries")
        
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_queries = queries[i:i + batch_size]
            
            batch_results = []
            for img, query in zip(batch_images, batch_queries):
                result = self.generate_answer(img, query)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def save_model(self, save_path: str):
        """Save model and processor"""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        logger.info(f"Medical generator saved to {save_path}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = MedicalAnswerGenerator()
    
    # Create dummy image for testing
    dummy_image = Image.new('RGB', (224, 224), color='white')
    
    # Test different types of medical queries
    test_queries = [
        "What abnormalities are visible in this chest X-ray?",
        "Describe the anatomical structures in this image",
        "Is this medical image normal or abnormal?",
        "What is the most likely diagnosis based on this image?"
    ]
    
    # Test answer generation
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = generator.generate_answer(dummy_image, query)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    # Test with mock retrieved context
    mock_context = [
        {
            'type': 'image',
            'score': 0.85,
            'metadata': {
                'condition': 'pneumonia',
                'findings': 'bilateral infiltrates'
            }
        },
        {
            'type': 'text',
            'score': 0.72,
            'metadata': {
                'text': 'Pneumonia typically presents with consolidation and air bronchograms on chest radiography.'
            }
        }
    ]
    
    print("\n--- Test with retrieved context ---")
    result_with_context = generator.generate_answer(
        dummy_image,
        "What condition does this chest X-ray show?",
        retrieved_context=mock_context
    )
    print(f"Answer with context: {result_with_context['answer']}")
    print(f"Confidence: {result_with_context['confidence']:.3f}")
    print(f"Context used: {result_with_context['retrieved_context_used']} items")