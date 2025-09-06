# Medical Multimodal RAG System

A medical-focused multimodal Retrieval-Augmented Generation system that combines medical image analysis with text-based knowledge retrieval to answer clinical questions.

## 🚀 NEW: Local RAG-Anything Integration

We've successfully integrated RAG-Anything with **local open-source models**, eliminating the need for API keys while providing enterprise-grade privacy and control.

### ✨ Key Features

**Original Medical RAG Components:**
- **Medical Image Understanding**: CLIP-based encoding with medical domain adaptations
- **Medical Text Processing**: Specialized preprocessing for medical terminology and abbreviations
- **Knowledge Retrieval**: FAISS-powered efficient similarity search for images and texts
- **Answer Generation**: BLIP-based multimodal answer generation with medical context
- **Modality Support**: Chest X-rays, CT scans, MRI, histology, and other medical images
- **Clinical Context**: Medical abbreviation expansion, anatomy standardization, and clinical disclaimers

**NEW: Local RAG-Anything Features:**
- 🔒 **Privacy-First**: All models run locally, no data sent to external APIs
- 🌐 **Enterprise Ready**: Full document processing (PDF, images, tables, equations)
- 🧠 **Local LLMs**: Mistral, Llama, Phi-3, and other open-source models
- 👁️ **Local Vision**: LLaVA, InternVL for image understanding
- 📊 **Local Embeddings**: BGE, E5 models for semantic search
- ⚕️ **Medical Specialization**: Domain-specific processing for medical documents
- 🎛️ **Flexible Presets**: "fast", "balanced", "quality", "medical" configurations
- 💾 **GPU Optimization**: 4-bit/8-bit quantization for memory efficiency

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM (16GB+ recommended)

### Super Fast Setup with uv ⚡ (Recommended)

**uv is 10-100x faster than pip for dependency management!**

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd muRAG
```

2. **One-command setup** (installs uv + dependencies + tests)
```bash
python setup.py
```

That's it! The setup script will:
- ✅ Install uv package manager
- ✅ Create virtual environment  
- ✅ Install all dependencies
- ✅ Set up development tools
- ✅ Run tests to verify everything works
- ✅ Install GPU support (if CUDA detected)

### Manual uv Setup (Alternative)

If you prefer manual control:

1. **Install uv**
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
pip install uv
```

2. **Create project and install dependencies**
```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the project
uv pip install -e .

# For GPU support
uv pip install -e .[gpu]

# For development
uv pip install -e .[dev]
```

### Traditional pip Setup (Slower)

If you must use pip:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt  # This will be much slower
```

### Basic Usage

#### 🔥 NEW: Local RAG-Anything (Recommended)

```python
from src.local_models import create_local_rag_anything

# Create local RAG system (no API keys needed!)
rag = create_local_rag_anything(
    working_dir="./my_rag_storage",
    model_preset="balanced"  # Uses Mistral-7B + LLaVA + BGE
)

# Initialize 
await rag.initialize()

# Process any document type
await rag.insert_file("medical_report.pdf")  # PDFs
await rag.insert_file("chest_xray.jpg")     # Images  
await rag.insert_directory("./medical_docs") # Batch processing

# Query with natural language
response = await rag.query("What does this report say about the patient's condition?")
print(response)

# Multimodal queries
multimodal_content = [
    {
        "type": "image", 
        "img_path": "chest_xray.jpg"
    },
    {
        "type": "table",
        "table_data": "Patient,Age,Condition\nJohn,65,Pneumonia"
    }
]

response = await rag.query(
    query="Analyze the chest X-ray and patient data",
    multimodal_content=multimodal_content
)
```

#### 🏥 Medical Specialized RAG

```python
from src.local_models import MedicalRAGAnything

# Medical domain specialization
medical_rag = MedicalRAGAnything(
    working_dir="./medical_rag_storage",
    model_preset="medical"  # Optimized for medical content
)

await medical_rag.initialize()

# Process medical documents with metadata
await medical_rag.insert_medical_document(
    file_path="patient_report.pdf",
    document_type="clinical_notes", 
    patient_id="12345"
)

# Medical queries with domain knowledge
response = await medical_rag.query("What is the differential diagnosis based on these symptoms?")
```

#### Original Medical RAG (Legacy)

```python
from src.medical_rag import MedicalMultimodalRAG
from PIL import Image

# Initialize the system
medical_rag = MedicalMultimodalRAG()

# Build knowledge base (with your medical data)
image_data = [
    {
        'image_path': 'path/to/chest_xray.jpg',
        'metadata': {
            'condition': 'pneumonia',
            'modality': 'chest_xray',
            'anatomy': 'chest',
            'description': 'Chest X-ray showing bilateral infiltrates'
        },
        'modality': 'chest_xray'
    }
]

text_data = [
    {
        'text': 'Pneumonia appears as areas of consolidation on chest X-rays...',
        'metadata': {
            'source': 'medical_textbook',
            'topic': 'respiratory_diseases'
        }
    }
]

medical_rag.build_knowledge_base(image_data, text_data)

# Query the system
image = Image.open('query_image.jpg')
result = medical_rag.query(
    image=image,
    question="What abnormalities are visible in this chest X-ray?",
    k_retrieve=5
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

## 🧪 Testing

### Local RAG-Anything Tests

```bash
# Quick model integration check
python test_local_rag_integration.py --quick

# Full integration test with model loading
python test_local_rag_integration.py --full

# Run comprehensive demo
python examples/local_rag_anything_demo.py --preset fast
```

### Original Medical RAG Tests

```bash
# With uv (recommended)
uv run python test_medical_rag.py

# Or if environment is activated
python test_medical_rag.py
```

**Local RAG-Anything Test Coverage:**
- ✅ Local LLM Models (Mistral, Llama, Phi-3)
- ✅ Local Vision Models (LLaVA, InternVL)
- ✅ Local Embedding Models (BGE, E5)
- ✅ RAG-Anything Integration
- ✅ Document Processing (PDF, images, tables)
- ✅ Multimodal Queries
- ✅ Medical Domain Specialization
- ✅ Model Presets and Configurations

**Original Medical RAG Test Coverage:**
- ✅ Medical Text Preprocessor
- ✅ Medical Image Encoder  
- ✅ Medical Knowledge Retriever
- ✅ Medical Answer Generator
- ✅ Integrated System
- ✅ Save/Load Functionality

## ⚡ uv Usage Guide

### Dependency Management
```bash
# Add new dependencies (much faster than pip install)
uv add torch torchvision transformers

# Add development dependencies
uv add --dev pytest black isort

# Add GPU support
uv add faiss-gpu --force-reinstall

# Remove dependencies
uv remove package-name

# Sync all dependencies
uv pip sync
```

### Running Commands
```bash
# Run any command in the virtual environment
uv run python test_medical_rag.py
uv run python -m pytest tests/
uv run jupyter notebook

# Run scripts defined in pyproject.toml
uv run medical-rag-test  # Runs test suite
uv run medical-rag-demo  # Starts demo (when implemented)
```

### Environment Management
```bash
# Create new environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install project in development mode
uv pip install -e .

# Export current environment
uv pip freeze > requirements.txt
```

## 📁 Project Structure

```
mmrag-cs6101/
├── src/
│   ├── local_models/              # 🔥 NEW: Local RAG-Anything Integration
│   │   ├── __init__.py
│   │   ├── local_llm_wrapper.py   # Local LLM models (Mistral, Llama, Phi-3)
│   │   ├── local_vision_wrapper.py # Local vision models (LLaVA, InternVL)
│   │   ├── local_embedding_wrapper.py # Local embeddings (BGE, E5)
│   │   └── local_rag_anything.py  # Complete RAG-Anything integration
│   ├── models/                    # Original Medical RAG
│   │   ├── medical_encoder.py      # CLIP-based medical image encoder
│   │   ├── medical_retriever.py    # FAISS-based knowledge retriever
│   │   └── medical_generator.py    # BLIP-based answer generator
│   ├── data/
│   │   └── medical_preprocessor.py # Medical text preprocessing
│   ├── training/                   # Training scripts (future)
│   ├── evaluation/                 # Evaluation metrics (future)
│   ├── demo/                       # Web demo interface (future)
│   └── medical_rag.py             # Main integrated system
├── RAG-Anything/                  # 🔥 RAG-Anything submodule
│   ├── raganything/
│   │   ├── config.py
│   │   ├── query.py
│   │   ├── modalprocessors.py
│   │   └── processor.py
│   └── examples/
├── examples/
│   └── local_rag_anything_demo.py # 🔥 Comprehensive local RAG demo
├── data/                          # Dataset storage
├── checkpoints/                   # Model weights
├── experiments/                   # Experiment logs
├── tests/                        # Unit tests
├── requirements.txt              # Dependencies  
├── test_medical_rag.py          # Original test suite
├── test_local_rag_integration.py # 🔥 Local RAG integration tests
└── README.md                    # This file
```

## 🏥 Medical Datasets

### Recommended Public Datasets

1. **VQA-Med** - Medical Visual Question Answering
   ```bash
   wget https://github.com/abachaa/VQA-Med-2019/archive/master.zip
   # 4,200 medical images with 15,292 QA pairs
   ```

2. **PathVQA** - Pathology VQA
   ```bash
   # 32,799 questions from 4,998 pathology images
   ```

3. **MIMIC-CXR** - Chest X-ray Reports
   ```bash
   # Requires PhysioNet credentialed access
   # 377,110 images, 227,835 reports
   ```

### Data Format

```python
# Image data format
image_data = [
    {
        'image_path': 'path/to/image.jpg',
        'metadata': {
            'modality': 'chest_xray',      # Image modality
            'condition': 'pneumonia',       # Medical condition
            'anatomy': 'chest',             # Anatomical region
            'findings': 'bilateral infiltrates',  # Medical findings
            'description': 'Detailed medical description'
        },
        'modality': 'chest_xray'  # For preprocessing
    }
]

# Text data format
text_data = [
    {
        'text': 'Medical knowledge text...',
        'metadata': {
            'source': 'medical_textbook',
            'topic': 'respiratory_diseases',
            'specialty': 'pulmonology'
        }
    }
]
```

## 🔧 Configuration

### Hardware Optimization

For **4x RTX 2080Ti** setup:
```bash
# Install GPU support with uv (much faster)
uv add faiss-gpu --force-reinstall

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

```python
# Distributed training configuration
medical_rag = MedicalMultimodalRAG(
    device="cuda",
    use_gpu_index=True  # Use GPU for FAISS
)

# Batch processing for large datasets
medical_rag.build_knowledge_base(
    image_data=your_image_data,
    text_data=your_text_data,
    batch_size=8  # Adjust based on GPU memory
)
```

For **RTX 5070Ti** development:
```bash
# Quick development environment setup
uv pip install -e .[dev]
```

```python
# Development and demo hosting
medical_rag = MedicalMultimodalRAG(
    device="cuda",
    cache_dir="./cache"
)
```

### 🎛️ Local Model Presets

The Local RAG-Anything system includes predefined model configurations:

```python
# Fast preset - for development and testing
rag = create_local_rag_anything(
    model_preset="fast",
    # Uses: Phi-3 + LLaVA-7B + BGE-small
    # Memory: ~4-6GB GPU
    # Speed: Fast inference
)

# Balanced preset - good quality/performance trade-off  
rag = create_local_rag_anything(
    model_preset="balanced", 
    # Uses: Mistral-7B + LLaVA-7B + BGE-base
    # Memory: ~8-12GB GPU
    # Speed: Moderate inference
)

# Quality preset - best results
rag = create_local_rag_anything(
    model_preset="quality",
    # Uses: Llama2-7B + LLaVA-13B + BGE-large
    # Memory: ~14-20GB GPU  
    # Speed: Slower but high quality
)

# Medical preset - optimized for medical documents
rag = create_local_rag_anything(
    model_preset="medical",
    # Medical-specific preprocessing
    # Expanded medical abbreviations
    # Clinical disclaimer injection
)
```

### 🔧 Custom Model Configuration

```python
# Custom configuration
rag = create_local_rag_anything(
    working_dir="./custom_rag",
    llm_model="mistralai/Mistral-7B-Instruct-v0.1",
    vision_model="llava-hf/llava-1.5-13b-hf", 
    embedding_model="BAAI/bge-large-en-v1.5",
    load_in_4bit=True,  # Enable quantization
    device="cuda",
    cache_dir="./model_cache"
)
```

### Medical Domain Adaptation

```python
# Custom medical terminology
from src.data.medical_preprocessor import MedicalTextPreprocessor

preprocessor = MedicalTextPreprocessor()

# Add custom abbreviations
preprocessor.medical_abbreviations.update({
    "MI": "myocardial infarction",
    "COPD": "chronic obstructive pulmonary disease"
})

# Process medical queries
processed_query = preprocessor.preprocess_medical_query(
    "Pt w/ SOB and chest pain, r/o MI"
)
```

## 📊 Evaluation Metrics

The system supports medical-specific evaluation:

- **Medical Term F1**: Accuracy of medical terminology
- **Anatomical Accuracy**: Correct anatomical region identification  
- **Clinical Relevance**: Relevance to clinical context
- **Retrieval Quality**: Quality of retrieved similar cases
- **Response Time**: System latency

```python
# Evaluate on medical datasets
from src.evaluation.medical_metrics import MedicalEvaluator

evaluator = MedicalEvaluator()
results = evaluator.evaluate_medical_accuracy(predictions, references)
```

## 🌐 Web Demo (Coming Soon)

```python
# Gradio-based medical interface
from src.demo.gradio_interface import MedicalRAGDemo

demo = MedicalRAGDemo(medical_rag)
interface = demo.create_interface()
interface.launch()
```

## 🧠 Model Architecture

### Components

1. **Medical Image Encoder** (`medical_encoder.py`)
   - Based on CLIP ViT-B/32
   - Medical-specific preprocessing (DICOM, windowing)
   - Modality-aware processing

2. **Knowledge Retriever** (`medical_retriever.py`)
   - FAISS-based similarity search
   - Supports both images and texts
   - Metadata filtering for medical attributes

3. **Answer Generator** (`medical_generator.py`)
   - Based on BLIP-large
   - Medical context integration
   - Clinical disclaimer injection

4. **Text Preprocessor** (`medical_preprocessor.py`)
   - Medical abbreviation expansion (200+ terms)
   - Anatomical term standardization
   - Medical entity extraction

### Pipeline

```
Medical Query + Image → 
  Text Preprocessing → 
    Multimodal Encoding → 
      Knowledge Retrieval → 
        Context-Aware Generation → 
          Medical Answer
```

## 🔬 Research Extensions

### Planned Features

1. **Multi-hop Reasoning**: Complex medical reasoning across multiple images
2. **Temporal Analysis**: Time-series medical image analysis  
3. **Domain Specialization**: Radiology, pathology, cardiology modules
4. **Clinical Decision Support**: Integration with EHR systems

### Training Pipeline

```python
# Future training implementation
from src.training.medical_trainer import MedicalRAGTrainer

trainer = MedicalRAGTrainer(medical_rag)
trainer.train_on_medical_data(
    vqa_med_data=vqa_med,
    path_vqa_data=path_vqa,
    mimic_cxr_data=mimic_data
)
```

## 📖 Citation

If you use this system in your research, please cite:

```bibtex
@software{medical_multimodal_rag_2024,
  title={Medical Multimodal RAG: A Specialized System for Clinical Question Answering},
  author={Your Team},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@domain.com

## 🔗 Related Work

- [MuRAG](https://arxiv.org/abs/2210.02928) - Original multimodal RAG
- [RagVL](https://arxiv.org/abs/2407.21439) - Advanced multimodal RAG with reranking
- [MMed-RAG](https://github.com/richard-peng-xia/MMed-RAG) - Medical domain RAG
- [VQA-Med](https://github.com/abachaa/VQA-Med-2019) - Medical VQA dataset

---

**⚕️ Medical Disclaimer**: This system is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult with qualified medical professionals for medical advice.