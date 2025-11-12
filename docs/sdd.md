# Software Design Document (SDD)
## MRAG-Bench Reproduction System

**Document Version:** 1.0
**Date:** September 27, 2024
**Author:** AI Engineer Architect
**Status:** Implementation Ready

---

## Executive Summary

This Software Design Document outlines the technical architecture for reproducing MRAG-Bench baseline results using a modular multimodal RAG system. The design prioritizes memory efficiency, performance optimization, and maintainability while operating within 16GB VRAM constraints to achieve 53-59% accuracy on perspective change scenarios.

**Key Architectural Decisions:**
- Modular component design with configurable model factory
- Tiered memory management with aggressive quantization
- Dedicated MRAG-Bench evaluation module
- Component-specific performance targets (retrieval: <5s, generation: <25s)

---

## 1. Project Architecture

### 1.1 High-Level System Design

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dataset       │    │    Retrieval     │    │   Generation    │
│   Interface     │──→ │    Pipeline      │──→ │    Pipeline     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ MRAG-Bench      │    │ CLIP ViT-B/32    │    │ LLaVA-1.5-7B    │
│ Data Loader     │    │ + Vector Store   │    │ + 4-bit Quant   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌──────────────────┐
                    │   Evaluation     │
                    │   Framework      │
                    └──────────────────┘
```

### 1.2 Core Components

**Dataset Interface Module (`src/dataset/`)**
- Unified data loading and preprocessing
- Perspective change scenario filtering
- Efficient batch processing support

**Retrieval Pipeline (`src/retrieval/`)**
- CLIP-based image embedding generation
- Vector similarity search with FAISS
- Top-k retrieval with configurable ranking

**Generation Pipeline (`src/generation/`)**
- LLaVA model integration with quantization
- Multimodal prompt construction
- Response generation and formatting

**Evaluation Framework (`src/evaluation/`)**
- MRAG-Bench methodology implementation
- Accuracy calculation and reporting
- Performance benchmarking tools

### 1.3 Data Flow Architecture

1. **Input Processing**: Query text + perspective change type filtering
2. **Image Retrieval**: CLIP embedding → similarity search → top-k images
3. **Context Assembly**: Retrieved images + query → multimodal prompt
4. **Generation**: LLaVA inference → medical answer generation
5. **Evaluation**: Answer validation → accuracy calculation → reporting

---

## 2. Software Stack

### 2.1 Core Technologies

**Programming Language**: Python 3.8+
- **Rationale**: Mature ML ecosystem, extensive library support, team expertise

**Deep Learning Framework**: PyTorch 2.0+
- **Rationale**: Native support for LLaVA/CLIP, advanced quantization features

**Model Libraries**:
- HuggingFace Transformers 4.30+ (model loading/inference)
- BitsAndBytes (4-bit quantization)
- CLIP (image-text embedding)

**Vector Database**: FAISS 1.7+
- **Rationale**: High-performance similarity search, GPU acceleration support

**Data Processing**:
- NumPy 1.21+ (numerical operations)
- Pillow 9.0+ (image processing)
- Pandas 1.5+ (dataset management)

### 2.2 Development Stack

**Package Management**: pip/conda
**Testing**: pytest 7.0+
**Code Quality**: black, flake8, mypy
**Documentation**: Sphinx with Google-style docstrings
**Version Control**: Git with conventional commits

### 2.3 Infrastructure Requirements

**Hardware**:
- RTX 5070Ti GPU (16GB VRAM)
- 32GB+ system RAM
- 1TB+ storage (dataset + models)

**Operating System**: Linux (Ubuntu 20.04+ recommended)
**CUDA**: 11.8+ for PyTorch compatibility

---

## 3. AI Models and Integration

### 3.1 Model Architecture

**Primary Vision-Language Model**: LLaVA-1.5-7B
- **Configuration**: 4-bit quantization (bnb-4bit)
- **Memory Footprint**: ~4-5GB VRAM quantized
- **Inference Time**: <25s per query (target)

**Image Retrieval Model**: CLIP ViT-B/32
- **Configuration**: Standard precision with memory optimization
- **Memory Footprint**: ~1GB VRAM
- **Inference Time**: <5s for embedding + retrieval (target)

### 3.2 Model Factory Pattern

```python
class ModelFactory:
    """Configurable model loading with fallback strategies."""

    @staticmethod
    def create_vlm(config: VLMConfig) -> VisionLanguageModel:
        # Primary: LLaVA-1.5-7B with 4-bit quantization
        # Fallback: LLaVA-1.5-13B with 8-bit if VRAM allows

    @staticmethod
    def create_retriever(config: RetrieverConfig) -> ImageRetriever:
        # Primary: CLIP ViT-B/32
        # Fallback: CLIP ViT-L/14 if performance insufficient
```

### 3.3 Memory Management Strategy

**Tiered Memory Optimization**:
1. **Model Level**: 4-bit quantization, gradient checkpointing
2. **Inference Level**: Dynamic model loading/unloading
3. **Data Level**: Streaming data processing, embedding caching
4. **System Level**: CUDA memory clearing, garbage collection

**Quantization Implementation**:
```python
# LLaVA quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 3.4 Model Pipeline Integration

**Sequential Pipeline Design**:
- Models loaded on-demand to minimize memory overlap
- Explicit memory cleanup between pipeline stages
- Fallback mechanisms for memory overflow scenarios

**Performance Monitoring**:
- Real-time VRAM usage tracking
- Inference timing measurement
- Automatic memory optimization triggers

---

## 4. Data Architecture

### 4.1 Dataset Organization

**MRAG-Bench Structure**:
```
data/
├── mrag_bench/
│   ├── images/           # 16,130 medical images
│   ├── questions/        # 1,353 perspective change questions
│   ├── annotations/      # Ground truth answers
│   └── metadata/         # Scenario type mappings
├── embeddings/
│   ├── clip_embeddings.npy    # Pre-computed image embeddings
│   └── faiss_index.bin        # FAISS similarity index
└── cache/
    ├── model_cache/      # Quantized model weights
    └── results_cache/    # Evaluation results
```

### 4.2 Data Processing Pipeline

**Preprocessing Components**:
1. **Image Standardization**: Resize to 224x224, normalize for CLIP
2. **Question Parsing**: Extract perspective change type metadata
3. **Embedding Generation**: Batch CLIP encoding with memory management
4. **Index Construction**: FAISS index building with GPU acceleration

**Streaming Data Access**:
```python
class MRAGDataset:
    """Memory-efficient dataset with streaming support."""

    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.perspective_filters = ['angle', 'partial', 'scope', 'occlusion']

    def stream_batches(self, scenario_type: str):
        # Yield batches without loading full dataset into memory
```

### 4.3 Embedding Storage Strategy

**Vector Database Design**:
- **Primary Index**: FAISS IVF with PQ compression
- **Embedding Dimension**: 512 (CLIP ViT-B/32)
- **Storage Format**: Memory-mapped arrays for efficient access
- **Retrieval Performance**: Sub-second similarity search

**Caching Strategy**:
- Persistent embedding cache to avoid recomputation
- LRU cache for frequently accessed model components
- Checkpointing for long-running evaluation processes

---

## 5. Implementation Plan

### 5.1 Technical Components

**Phase 1: Core Infrastructure (Weeks 1-2)**

```python
# Key interfaces to implement
class DatasetInterface:
    """Unified dataset access with perspective filtering."""
    def load_scenario(self, scenario_type: str) -> Iterator[Sample]
    def get_retrieval_corpus(self) -> List[Image]
    def preprocess_batch(self, samples: List[Sample]) -> BatchData

class RetrievalPipeline:
    """CLIP-based image retrieval system."""
    def encode_images(self, images: List[Image]) -> np.ndarray
    def build_index(self, embeddings: np.ndarray) -> FAISSIndex
    def retrieve_similar(self, query: str, k: int) -> List[RetrievalResult]

class GenerationPipeline:
    """LLaVA-based answer generation."""
    def load_model(self, config: ModelConfig) -> VLMModel
    def generate_answer(self, context: MultimodalContext) -> str
    def clear_memory(self) -> None
```

**Phase 2: Integration & Optimization (Weeks 3-4)**

```python
class MRAGBenchEvaluator:
    """Complete evaluation pipeline."""
    def __init__(self, config: EvaluationConfig):
        self.retriever = RetrievalPipeline(config.retrieval)
        self.generator = GenerationPipeline(config.generation)
        self.dataset = DatasetInterface(config.dataset)

    def evaluate_scenario(self, scenario: str) -> EvaluationResults:
        # End-to-end evaluation with memory management

    def compute_accuracy(self, predictions: List[str],
                        ground_truth: List[str]) -> float:
        # MRAG-Bench accuracy calculation
```

**Phase 3: Validation & Optimization (Weeks 5-6)**

- Comprehensive evaluation on all perspective change scenarios
- Performance profiling and optimization
- Accuracy validation against 53-59% baseline target

### 5.2 Configuration Management

**Configuration Structure**:
```yaml
# config/mrag_bench.yaml
model:
  vlm:
    name: "llava-hf/llava-1.5-7b-hf"
    quantization: "4bit"
    max_memory: "14GB"
  retriever:
    name: "openai/clip-vit-base-patch32"
    embedding_dim: 512

evaluation:
  scenarios: ["angle", "partial", "scope", "occlusion"]
  batch_size: 4
  top_k_retrieval: 5
  max_generation_length: 512

performance:
  memory_limit: "16GB"
  retrieval_timeout: 5
  generation_timeout: 25
```

### 5.3 Testing Strategy

**Unit Testing**:
- Component-level functionality validation
- Memory usage testing for individual modules
- Model loading/unloading verification

**Integration Testing**:
- End-to-end pipeline validation
- Performance benchmarking under memory constraints
- Accuracy testing on sample datasets

**Performance Testing**:
- VRAM usage monitoring throughout evaluation
- Inference timing validation against targets
- Stress testing with full dataset evaluation

### 5.4 Error Handling & Recovery

**Memory Management**:
```python
class MemoryManager:
    """Automatic memory management and recovery."""

    def __init__(self, max_vram_gb: float = 15.0):
        self.max_vram = max_vram_gb * 1024**3

    def check_memory(self) -> bool:
        # Monitor VRAM usage

    def emergency_cleanup(self) -> None:
        # Aggressive memory clearing for recovery

    @contextmanager
    def memory_guard(self):
        # Automatic cleanup on exceptions
```

**Fault Tolerance**:
- Graceful degradation on memory overflow
- Automatic retry with reduced batch sizes
- Comprehensive logging for debugging

### 5.5 Performance Targets

**Component-Specific Targets**:
- **Image Retrieval**: <5 seconds per query
- **Answer Generation**: <25 seconds per query
- **Total Pipeline**: <30 seconds per query
- **Memory Usage**: ≤15GB VRAM (1GB buffer)
- **Accuracy**: 53-59% on perspective change scenarios

**System Monitoring**:
- Real-time performance dashboards
- Memory usage alerts and automatic optimization
- Accuracy tracking throughout evaluation

---

## 6. Security and Compliance

### 6.1 Data Privacy
- Local processing only (no external API calls)
- Secure handling of medical imaging data
- Compliance with research data usage policies

### 6.2 Model Security
- Verified model checksums from trusted sources
- Isolated model execution environment
- Comprehensive logging for audit trails

### 6.3 Reproducibility
- Deterministic evaluation with fixed random seeds
- Version pinning for all dependencies
- Comprehensive environment documentation

---

## 7. Deployment and Operations

### 7.1 Environment Setup
```bash
# Production environment setup
conda create -n mrag-bench python=3.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install faiss-gpu pillow pandas numpy
```

### 7.2 Model Deployment
- Automated model downloading and caching
- Quantization preprocessing for faster loading
- Health checks for model initialization

### 7.3 Monitoring and Logging
- Structured logging with performance metrics
- Error tracking and alerting
- Resource utilization monitoring

---

## 8. Future Considerations

### 8.1 Scalability
- Multi-GPU support for larger models
- Distributed evaluation across multiple nodes
- Model serving optimization for production

### 8.2 Extensibility
- Plugin architecture for new retrieval methods
- Configurable evaluation metrics
- Support for additional multimodal datasets

### 8.3 Research Integration
- Easy integration with existing research workflows
- Comprehensive API for external tool integration
- Standardized output formats for analysis tools

---

**Implementation Checklist:**
- [ ] Core dataset interface implementation
- [ ] CLIP retrieval pipeline with FAISS
- [ ] LLaVA generation pipeline with quantization
- [ ] Memory management and optimization
- [ ] MRAG-Bench evaluation framework
- [ ] Configuration and testing infrastructure
- [ ] Performance validation and optimization
- [ ] Documentation and deployment guides

---

*This SDD serves as the technical blueprint for implementing the MRAG-Bench reproduction system within the specified 4-6 week timeline and 16GB VRAM constraints.*