"""
Medical Text Preprocessor for handling medical terminology, abbreviations, and domain-specific text processing
"""

import re
import string
from typing import Dict, List, Set, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MedicalTextPreprocessor:
    """
    Preprocessor for medical text with domain-specific knowledge
    """
    
    def __init__(self):
        """Initialize medical text preprocessor"""
        self.medical_abbreviations = self._load_medical_abbreviations()
        self.anatomy_terms = self._load_anatomy_terms()
        self.medical_conditions = self._load_medical_conditions()
        self.stop_words = self._load_medical_stop_words()
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("Medical text preprocessor initialized")
    
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """Load medical abbreviations dictionary"""
        return {
            # Cardiovascular
            "MI": "myocardial infarction",
            "CHF": "congestive heart failure",
            "HTN": "hypertension", 
            "CAD": "coronary artery disease",
            "DVT": "deep vein thrombosis",
            "PE": "pulmonary embolism",
            "AF": "atrial fibrillation",
            "VF": "ventricular fibrillation",
            "EKG": "electrocardiogram",
            "ECG": "electrocardiogram",
            
            # Respiratory
            "COPD": "chronic obstructive pulmonary disease",
            "SOB": "shortness of breath",
            "DOE": "dyspnea on exertion",
            "URI": "upper respiratory infection",
            "ARDS": "acute respiratory distress syndrome",
            "OSA": "obstructive sleep apnea",
            
            # Endocrine
            "DM": "diabetes mellitus",
            "T1DM": "type 1 diabetes mellitus",
            "T2DM": "type 2 diabetes mellitus",
            "DKA": "diabetic ketoacidosis",
            "HbA1c": "hemoglobin A1c",
            "TSH": "thyroid stimulating hormone",
            
            # Gastrointestinal
            "GERD": "gastroesophageal reflux disease",
            "IBD": "inflammatory bowel disease",
            "IBS": "irritable bowel syndrome",
            "GI": "gastrointestinal",
            "N/V": "nausea and vomiting",
            
            # Neurological
            "CVA": "cerebrovascular accident",
            "TIA": "transient ischemic attack",
            "MS": "multiple sclerosis",
            "ALS": "amyotrophic lateral sclerosis",
            "CNS": "central nervous system",
            "PNS": "peripheral nervous system",
            
            # Oncology
            "CA": "cancer",
            "mets": "metastases",
            "chemo": "chemotherapy",
            "RT": "radiation therapy",
            "bx": "biopsy",
            
            # Laboratory
            "CBC": "complete blood count",
            "BMP": "basic metabolic panel",
            "CMP": "comprehensive metabolic panel",
            "PT": "prothrombin time",
            "PTT": "partial thromboplastin time",
            "INR": "international normalized ratio",
            "ESR": "erythrocyte sedimentation rate",
            "CRP": "C-reactive protein",
            
            # General
            "Hx": "history",
            "Dx": "diagnosis",
            "Tx": "treatment",
            "Rx": "prescription",
            "Sx": "symptoms",
            "Pt": "patient",
            "yo": "year old",
            "y/o": "year old",
            "w/": "with",
            "w/o": "without",
            "c/w": "consistent with",
            "s/p": "status post",
            "r/o": "rule out"
        }
    
    def _load_anatomy_terms(self) -> Dict[str, List[str]]:
        """Load anatomical term mappings and synonyms"""
        return {
            "chest": ["thorax", "thoracic", "pulmonary", "lung", "cardiac", "heart"],
            "abdomen": ["abdominal", "gastric", "hepatic", "pancreatic", "splenic", "renal"],
            "brain": ["cerebral", "neural", "cranial", "intracranial", "neurologic"],
            "spine": ["spinal", "vertebral", "cervical", "thoracic", "lumbar", "sacral"],
            "pelvis": ["pelvic", "hip", "sacroiliac", "pubic"],
            "extremities": ["arm", "leg", "hand", "foot", "shoulder", "knee", "ankle", "wrist"],
            "head": ["cranium", "skull", "facial", "orbital", "nasal", "oral"],
            "neck": ["cervical", "thyroid", "carotid", "jugular"]
        }
    
    def _load_medical_conditions(self) -> Set[str]:
        """Load common medical conditions"""
        return {
            "pneumonia", "bronchitis", "asthma", "emphysema",
            "hypertension", "hypotension", "arrhythmia", "tachycardia", "bradycardia",
            "diabetes", "hypoglycemia", "hyperglycemia", 
            "fracture", "dislocation", "sprain", "strain",
            "infection", "inflammation", "necrosis", "ischemia",
            "tumor", "mass", "lesion", "nodule", "cyst",
            "edema", "effusion", "hemorrhage", "hematoma",
            "stenosis", "occlusion", "thrombosis", "embolism"
        }
    
    def _load_medical_stop_words(self) -> Set[str]:
        """Load medical-specific stop words to preserve"""
        # These are typically removed in general NLP but important in medical context
        return {
            "no", "not", "without", "absent", "negative", "normal", 
            "positive", "present", "mild", "moderate", "severe",
            "acute", "chronic", "stable", "unstable"
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient processing"""
        # Pattern for medical abbreviations (case-insensitive)
        abbrev_pattern = "|".join(re.escape(abbrev) for abbrev in self.medical_abbreviations.keys())
        self.abbrev_regex = re.compile(f"\\b({abbrev_pattern})\\b", re.IGNORECASE)
        
        # Pattern for measurements and values
        self.measurement_regex = re.compile(r'\b\d+(\.\d+)?\s*(mg|g|kg|ml|l|cm|mm|m|%|degrees?)\b', re.IGNORECASE)
        
        # Pattern for medical identifiers
        self.medical_id_regex = re.compile(r'\b(patient|case|study)\s*#?\s*\d+\b', re.IGNORECASE)
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations in text
        
        Args:
            text: Input medical text
            
        Returns:
            Text with expanded abbreviations
        """
        def replace_abbrev(match):
            abbrev = match.group(1)
            # Preserve case of original abbreviation
            expanded = self.medical_abbreviations.get(abbrev.upper(), abbrev)
            if abbrev.isupper():
                return expanded.upper()
            elif abbrev[0].isupper():
                return expanded.capitalize()
            else:
                return expanded.lower()
        
        return self.abbrev_regex.sub(replace_abbrev, text)
    
    def standardize_anatomy_terms(self, text: str) -> str:
        """
        Standardize anatomical terminology
        
        Args:
            text: Input text
            
        Returns:
            Text with standardized anatomy terms
        """
        standardized_text = text.lower()
        
        for standard_term, synonyms in self.anatomy_terms.items():
            for synonym in synonyms:
                # Replace synonym with standard term
                pattern = r'\b' + re.escape(synonym) + r'\b'
                standardized_text = re.sub(pattern, standard_term, standardized_text, flags=re.IGNORECASE)
        
        return standardized_text
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text
        
        Args:
            text: Input medical text
            
        Returns:
            Dictionary with extracted entities by category
        """
        entities = {
            'conditions': [],
            'measurements': [],
            'anatomy': [],
            'abbreviations': []
        }
        
        # Extract conditions
        words = text.lower().split()
        for word in words:
            clean_word = word.strip(string.punctuation)
            if clean_word in self.medical_conditions:
                entities['conditions'].append(clean_word)
        
        # Extract measurements
        measurements = self.measurement_regex.findall(text)
        entities['measurements'] = [f"{val}{unit}" for val, unit in measurements]
        
        # Extract anatomy terms
        for standard_term, synonyms in self.anatomy_terms.items():
            all_terms = [standard_term] + synonyms
            for term in all_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    if standard_term not in entities['anatomy']:
                        entities['anatomy'].append(standard_term)
        
        # Extract abbreviations
        abbrev_matches = self.abbrev_regex.findall(text)
        entities['abbreviations'] = list(set(abbrev_matches))
        
        return entities
    
    def preprocess_medical_query(self, query: str) -> str:
        """
        Comprehensive preprocessing for medical queries
        
        Args:
            query: Raw medical query
            
        Returns:
            Preprocessed query
        """
        # Remove extra whitespace
        processed_query = re.sub(r'\s+', ' ', query.strip())
        
        # Expand abbreviations
        processed_query = self.expand_abbreviations(processed_query)
        
        # Standardize anatomy terms
        processed_query = self.standardize_anatomy_terms(processed_query)
        
        # Normalize punctuation
        processed_query = re.sub(r'[^\w\s]', ' ', processed_query)
        processed_query = re.sub(r'\s+', ' ', processed_query)
        
        return processed_query.strip()
    
    def preprocess_medical_document(self, document: str) -> Dict[str, str]:
        """
        Preprocess medical document with detailed processing
        
        Args:
            document: Raw medical document
            
        Returns:
            Dictionary with processed text and extracted entities
        """
        # Basic preprocessing
        processed_text = self.preprocess_medical_query(document)
        
        # Extract entities
        entities = self.extract_medical_entities(document)
        
        # Create enhanced text with entity information
        enhanced_text = processed_text
        
        # Add entity context
        if entities['conditions']:
            enhanced_text += f" CONDITIONS: {' '.join(entities['conditions'])}"
        if entities['anatomy']:
            enhanced_text += f" ANATOMY: {' '.join(entities['anatomy'])}"
        
        return {
            'processed_text': processed_text,
            'enhanced_text': enhanced_text,
            'entities': entities,
            'original_length': len(document),
            'processed_length': len(processed_text)
        }
    
    def create_medical_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract key medical terms from text
        
        Args:
            text: Input medical text
            top_k: Number of top keywords to return
            
        Returns:
            List of medical keywords
        """
        entities = self.extract_medical_entities(text)
        
        # Combine all medical entities
        keywords = []
        keywords.extend(entities['conditions'])
        keywords.extend(entities['anatomy'])
        keywords.extend([self.medical_abbreviations.get(abbrev.upper(), abbrev) 
                        for abbrev in entities['abbreviations']])
        
        # Remove duplicates and return top-k
        unique_keywords = list(set(keywords))
        return unique_keywords[:top_k]
    
    def compute_medical_similarity(self, text1: str, text2: str) -> float:
        """
        Compute medical domain similarity between two texts
        
        Args:
            text1: First medical text
            text2: Second medical text
            
        Returns:
            Similarity score between 0 and 1
        """
        entities1 = self.extract_medical_entities(text1)
        entities2 = self.extract_medical_entities(text2)
        
        # Calculate overlap in different entity types
        condition_overlap = len(set(entities1['conditions']) & set(entities2['conditions']))
        anatomy_overlap = len(set(entities1['anatomy']) & set(entities2['anatomy']))
        abbrev_overlap = len(set(entities1['abbreviations']) & set(entities2['abbreviations']))
        
        # Total entities
        total_entities1 = sum(len(v) for v in entities1.values())
        total_entities2 = sum(len(v) for v in entities2.values())
        
        if total_entities1 == 0 and total_entities2 == 0:
            return 0.0
        
        # Weighted similarity
        total_overlap = condition_overlap * 2 + anatomy_overlap + abbrev_overlap
        max_entities = max(total_entities1, total_entities2)
        
        return min(total_overlap / max_entities, 1.0) if max_entities > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize preprocessor
    preprocessor = MedicalTextPreprocessor()
    
    # Test abbreviation expansion
    test_text = "Pt w/ MI and CHF, s/p CABG. EKG shows AF."
    expanded = preprocessor.expand_abbreviations(test_text)
    print(f"Original: {test_text}")
    print(f"Expanded: {expanded}")
    
    # Test entity extraction
    medical_text = "Patient presents with chest pain, elevated troponins, and EKG changes consistent with acute MI."
    entities = preprocessor.extract_medical_entities(medical_text)
    print(f"\nEntities: {entities}")
    
    # Test query preprocessing
    query = "Show me chest X-rays with pneumonia or lung infection"
    processed_query = preprocessor.preprocess_medical_query(query)
    print(f"\nOriginal query: {query}")
    print(f"Processed query: {processed_query}")
    
    # Test document preprocessing
    document = "65 yo male w/ Hx of CAD presents w/ SOB and chest pain. CXR shows bilateral infiltrates c/w pneumonia."
    processed_doc = preprocessor.preprocess_medical_document(document)
    print(f"\nProcessed document: {processed_doc}")
    
    # Test medical keywords
    keywords = preprocessor.create_medical_keywords(document)
    print(f"\nMedical keywords: {keywords}")