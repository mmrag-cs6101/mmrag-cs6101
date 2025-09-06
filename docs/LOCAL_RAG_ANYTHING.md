# Local RAG-Anything Integration Guide

This document provides detailed information about the Local RAG-Anything integration, which replaces API-based models with local open-source alternatives.

## 🎯 Overview

Our Local RAG-Anything integration provides:
- **Privacy-First**: All models run locally, no data leaves your system
- **Enterprise Ready**: Full document processing without external dependencies  
- **Medical Specialization**: Domain-specific processing for medical documents
- **Flexible Configuration**: Multiple model presets and custom configurations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Local RAG-Anything                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Local     │  │   Local     │  │    Local            │ │
│  │   LLM       │  │   Vision    │  │    Embeddings       │ │
│  │ (Mistral,   │  │ (LLaVA,     │  │  (BGE, E5)          │ │
│  │  Llama,     │  │  InternVL)  │  │                     │ │
│  │  Phi-3)     │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    RAG-Anything Core                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Document   │  │ Multimodal  │  │    Knowledge        │ │
│  │  Parser     │  │ Processor   │  │    Graph            │ │
│  │ (PDF, IMG,  │  │ (Images,    │  │   (LightRAG)        │ │
│  │  Tables)    │  │  Tables)    │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Medical RAG System                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Medical    │  │  Medical    │  │   Medical           │ │
│  │  Text       │  │  Image      │  │   Answer            │ │
│  │ Processor   │  │  Encoder    │  │  Generator          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

1. **Install RAG-Anything Dependencies**:
   ```bash
   pip install lightrag-hku mineru[core] tqdm
   ```

2. **Install RAG-Anything**:
   ```bash
   cd RAG-Anything
   pip install -e .
   ```

3. **Install Additional Dependencies**:
   ```bash
   pip install sentence-transformers transformers torch torchvision
   ```

### Basic Usage

```python
import asyncio
from src.local_models import create_local_rag_anything

async def main():
    # Create RAG system
    rag = create_local_rag_anything(
        working_dir="./my_rag_storage",
        model_preset="fast"  # Good for testing
    )
    
    # Initialize
    await rag.initialize()
    
    # Process documents
    await rag.insert_file("document.pdf")
    
    # Query
    response = await rag.query("What does this document discuss?")
    print(response)

# Run
asyncio.run(main())
```

## 🎛️ Model Presets

### Fast Preset
**Best for**: Development, testing, resource-constrained environments
```python
rag = create_local_rag_anything(model_preset="fast")
```
- **LLM**: Qwen2-1.5B-Instruct (fast and unrestricted)
- **Vision**: LLaVA-1.5-7B 
- **Embeddings**: BGE-small (384 dimensions)
- **Memory**: ~2-4GB GPU
- **Speed**: Very fast inference
- **No Authentication**: All models are publicly available

### Balanced Preset (Default)
**Best for**: Production use with good quality/performance trade-off
```python
rag = create_local_rag_anything(model_preset="balanced")
```
- **LLM**: Gemma-2B-IT (Google, unrestricted)
- **Vision**: LLaVA-1.5-7B
- **Embeddings**: BGE-base (768 dimensions)
- **Memory**: ~4-8GB GPU
- **Speed**: Good balance of speed and quality
- **No Authentication**: All models are publicly available

### Quality Preset
**Best for**: High-quality results, research applications
```python
rag = create_local_rag_anything(model_preset="quality")
```
- **LLM**: Mistral-7B-Instruct-v0.3 (unrestricted version)
- **Vision**: LLaVA-1.5-7B
- **Embeddings**: BGE-large (1024 dimensions)
- **Memory**: ~8-14GB GPU
- **Speed**: Slower but high quality
- **No Authentication**: Uses unrestricted Mistral version

### Medical Preset
**Best for**: Medical documents and clinical applications
```python
rag = create_local_rag_anything(model_preset="medical")
# OR
from src.local_models import MedicalRAGAnything
rag = MedicalRAGAnything()
```
- **LLM**: Gemma-2B-IT (medical-optimized and unrestricted)
- **Vision**: LLaVA-1.5-7B
- **Embeddings**: BGE-base with medical preprocessing
- **Features**: Medical abbreviation expansion, clinical disclaimers
- **No Authentication**: All models are publicly available

## 🔧 Custom Configuration

### Advanced Model Configuration

```python
rag = create_local_rag_anything(
    working_dir="./custom_rag",
    
    # Model selection
    llm_model="mistralai/Mistral-7B-Instruct-v0.1",
    vision_model="llava-hf/llava-1.5-13b-hf",
    embedding_model="BAAI/bge-large-en-v1.5",
    
    # Hardware configuration
    device="cuda",  # or "cpu", "auto"
    load_in_4bit=True,  # Enable 4-bit quantization
    load_in_8bit=False,  # Enable 8-bit quantization
    cache_dir="./model_cache",
    
    # RAG configuration
    rag_config=RAGAnythingConfig(
        working_dir="./custom_rag",
        context_window=2,
        max_context_tokens=4000,
        enable_image_processing=True,
        enable_table_processing=True
    )
)
```

### Using Custom Models

```python
from src.local_models import LocalLLMManager, create_local_rag_anything

# Create custom LLM configuration
custom_config = {
    "model_name": "microsoft/DialoGPT-medium",  # Any HuggingFace model
    "max_length": 1024,
    "temperature": 0.8,
    "load_in_4bit": False
}

rag = create_local_rag_anything(
    llm_model="custom",
    custom_config=custom_config
)
```

## 📊 Document Processing

### Supported File Types

- **PDF**: Full text and image extraction
- **Images**: JPG, PNG, BMP, TIFF, WebP
- **Office**: DOC, DOCX, PPT, PPTX, XLS, XLSX
- **Text**: TXT, MD (Markdown)

### Processing Examples

```python
# Single file processing
result = await rag.insert_file("medical_report.pdf")

# Batch directory processing
result = await rag.insert_directory("./medical_docs", recursive=True)

# Medical document with metadata
await rag.insert_medical_document(
    file_path="patient_chart.pdf",
    document_type="clinical_notes",
    patient_id="PATIENT_001"
)
```

## 🔍 Querying

### Text Queries

```python
# Simple text query
response = await rag.query("What is the main finding?")

# Query with specific mode
response = await rag.query("Summarize the key points", mode="global")
```

### Multimodal Queries

```python
# Query with images and tables
multimodal_content = [
    {
        "type": "image",
        "img_path": "chest_xray.jpg"
    },
    {
        "type": "table", 
        "table_data": "Name,Age,Condition\nJohn,65,Pneumonia",
        "table_caption": ["Patient data"]
    },
    {
        "type": "equation",
        "text": "SaO2 = (HbO2 / (HbO2 + Hb)) * 100",
        "text_format": "LaTeX"
    }
]

response = await rag.query(
    query="Analyze the chest X-ray in context of patient data",
    multimodal_content=multimodal_content
)
```

### Query Modes

- **`local`**: Search within local document chunks
- **`global`**: Search across the entire knowledge graph
- **`hybrid`**: Combination of local and global search  
- **`naive`**: Simple similarity search
- **`mix`**: Adaptive mode (default)
- **`bypass`**: Direct LLM without retrieval

## 🏥 Medical Specialization

### Medical Text Processing

The medical mode includes:

```python
# Medical abbreviation expansion
"Pt w/ SOB and chest pain, r/o MI" 
→ "Patient with shortness of breath and chest pain, rule out myocardial infarction"

# Anatomical term standardization
"lung" → "pulmonary"
"heart" → "cardiac"

# Medical entity extraction
# Extracts conditions, procedures, medications, anatomy
```

### Medical Query Enhancement

```python
# Medical-specific query processing
medical_rag = MedicalRAGAnything()
await medical_rag.initialize()

# Enhanced with medical context
response = await medical_rag.query(
    "What is the differential diagnosis for chest pain with ST elevation?"
)
# Automatically adds clinical disclaimers and medical uncertainty language
```

## 🚀 Performance Optimization

### Memory Optimization

```python
# Enable quantization for memory savings
rag = create_local_rag_anything(
    model_preset="balanced",
    load_in_4bit=True,  # Reduces memory by ~75%
    custom_config={
        "batch_size": 4,  # Reduce batch size
        "max_seq_length": 512  # Limit sequence length
    }
)
```

### GPU Utilization

```python
# Multi-GPU setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

rag = create_local_rag_anything(
    device="cuda",
    custom_config={"device_map": "auto"}  # Automatic GPU assignment
)
```

### Batch Processing

```python
# Process multiple files efficiently
file_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

for file_path in file_paths:
    await rag.insert_file(file_path)

# Or batch directory processing
await rag.insert_directory("./documents", recursive=True)
```

## 🧪 Testing and Validation

### Running Tests

```bash
# Quick integration test
python test_local_rag_integration.py --quick

# Full integration test with model loading
python test_local_rag_integration.py --full

# Run comprehensive demo
python examples/local_rag_anything_demo.py --preset fast
```

### Custom Testing

```python
# Test custom configuration
from src.local_models import LocalLLMManager

# Test LLM functionality
manager = LocalLLMManager()
llm = manager.load_model("phi-3")
response = await llm.agenerate("What is AI?")
print(response)
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   ```python
   # Enable quantization
   rag = create_local_rag_anything(
       model_preset="fast",  # Use smaller models
       load_in_4bit=True     # Enable quantization
   )
   ```

2. **Slow Inference**:
   ```python
   # Use GPU and smaller models
   rag = create_local_rag_anything(
       model_preset="fast",
       device="cuda",
       custom_config={"batch_size": 1}
   )
   ```

3. **Model Loading Errors**:
   ```bash
   # Install missing dependencies
   pip install sentence-transformers transformers torch
   ```

### Performance Monitoring

```python
# Get model information
info = rag.get_model_info()
print(f"Models loaded: {info}")

# Memory cleanup
rag.cleanup()  # Unload all models
```

## 🔗 Integration with Existing Medical RAG

```python
# Combine with original Medical RAG system
from src.local_models import create_local_rag_anything
from src.medical_rag import MedicalMultimodalRAG

# Use Local RAG-Anything for document processing
local_rag = create_local_rag_anything(model_preset="medical")
await local_rag.insert_directory("./medical_docs")

# Use original Medical RAG for specialized medical queries
medical_rag = MedicalMultimodalRAG()
medical_rag.build_knowledge_base(image_data, text_data)

# Query both systems as needed
local_response = await local_rag.query("Process this medical report")
medical_response = medical_rag.query(image, "Analyze this X-ray")
```

## 📈 Scalability Considerations

### Large Document Collections

```python
# For large document collections (1000+ documents)
rag = create_local_rag_anything(
    model_preset="balanced",
    rag_config=RAGAnythingConfig(
        max_concurrent_files=2,  # Limit concurrent processing
        context_window=1,        # Reduce context window
        max_context_tokens=2000  # Limit context size
    )
)
```

### Production Deployment

```python
# Production-ready configuration
rag = create_local_rag_anything(
    working_dir="/data/rag_storage",  # Persistent storage
    model_preset="balanced",
    device="cuda",
    cache_dir="/data/model_cache",   # Persistent model cache
    rag_config=RAGAnythingConfig(
        enable_llm_cache=True,       # Enable LLM response cache
        max_concurrent_files=4       # Parallel processing
    )
)
```

This integration provides a complete, privacy-preserving alternative to API-based RAG systems while maintaining full compatibility with medical domain requirements.