# Multimodal RAG Implementation Plan

A comprehensive guide for implementing and extending multimodal Retrieval-Augmented Generation systems, with focus on replicating and improving upon recent research.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Available Implementations](#available-implementations)
3. [Recommended Starting Strategy](#recommended-starting-strategy)
4. [Environment Setup](#environment-setup)
5. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
6. [Code Examples](#code-examples)
7. [Research Extensions](#research-extensions)
8. [Resources and References](#resources-and-references)

## 🎯 Overview

This project focuses on implementing multimodal RAG systems, starting with replication of existing work and extending to novel contributions. We'll build upon the foundational MuRAG paper while leveraging more recent advances like RagVL.

**Key Papers:**
- **MuRAG (2022)**: "Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text" - Foundational work
- **RagVL (2024)**: "MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training" - State-of-the-art with available code

## 🚀 Available Implementations

### 1. RagVL (2024) - **HIGHLY RECOMMENDED** ⭐⭐⭐

**Repository**: [IDEA-FinAI/RagVL](https://github.com/IDEA-FinAI/RagVL)

**Why Choose RagVL:**
- ✅ Complete PyTorch implementation with datasets and checkpoints
- ✅ State-of-the-art results on WebQA and MultimodalQA
- ✅ Addresses multi-granularity noisy correspondence (MNC) problem
- ✅ Supports multiple MLLMs (LLaVA, Qwen-VL, mPLUG-Owl2, InternVL2)
- ✅ Ready-to-run evaluation scripts
- ✅ Built on MuRAG concepts with 2024 improvements

**Key Innovations:**
- Knowledge-enhanced reranking using MLLMs
- Noise-injected training at data and token levels
- Adaptive threshold filtering
- Multi-stage training pipeline

### 2. MMed-RAG (2024) - Medical Domain ⭐⭐⭐

**Repository**: [richard-peng-xia/MMed-RAG](https://github.com/richard-peng-xia/MMed-RAG)

**Features:**
- ICLR'25 accepted paper
- Boosts medical VLM factuality by up to 43.8%
- Domain-aware retrieval mechanism
- Addresses over-reliance on retrieved contexts

### 3. Lightweight Educational Implementations ⭐⭐

- **[13331112522/m-rag](https://github.com/13331112522/m-rag)**: 300-line multimodal RAG
- **[CornelliusYW/Multimodal-RAG-Implementation](https://github.com/CornelliusYW/Multimodal-RAG-Implementation)**: CLIP + Whisper + ChromaDB
- **[Azure-Samples/multimodal-rag-code-execution](https://github.com/Azure-Samples/multimodal-rag-code-execution)**: Enterprise-ready implementation
- **[LangChain Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb)**: Official tutorial

## 🎯 Recommended Starting Strategy

### Option 1: Start with RagVL (Recommended for Research)

Best for: Academic projects, research contributions, state-of-the-art results

### Option 2: Start Simple, Scale Up

Best for: Learning fundamentals, quick prototyping, understanding core concepts

## 🛠️ Environment Setup

### Using uv (Recommended - 10x faster than pip)

```bash
# Install uv if you haven't already
pip install uv

# Create project and virtual environment
git clone https://github.com/IDEA-FinAI/RagVL.git  # or your chosen repo
cd RagVL
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (much faster than pip)
uv add -r requirements.txt

# For GPU support
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Using venv (Alternative)

```bash
python -m venv multimodal-rag-env
source multimodal-rag-env/bin/activate  # On Windows: multimodal-rag-env\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Project Structure Setup

```bash
mkdir -p {src,data,experiments,tests,configs,scripts}
touch src/__init__.py tests/__init__.py

# Recommended structure:
# multimodal-rag/
# ├── src/
# │   ├── models/
# │   ├── data/
# │   ├── training/
# │   └── evaluation/
# ├── data/
# ├── experiments/
# ├── tests/
# ├── configs/
# ├── scripts/
# ├── requirements.txt
# └── README.md
```

## 📅 Phase-by-Phase Implementation

### Phase 1: Setup & Quick Start (Week 1)

#### 1.1 Environment Verification

```bash
# Create test script
cat > test_environment.py << 'EOF'
import torch
import transformers
import PIL
from transformers import CLIPModel, CLIPProcessor
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print("✅ Environment ready!")
EOF

python test_environment.py
```

#### 1.2 RagVL Setup

```bash
# Clone and setup RagVL
git clone https://github.com/IDEA-FinAI/RagVL.git
cd RagVL
uv venv && source .venv/bin/activate
uv add -r requirements.txt

# Download datasets and checkpoints (follow their instructions)
# Datasets: WebQA, MultimodalQA
# Checkpoints: Pre-trained models for different MLLMs
```

#### 1.3 Quick Evaluation

```bash
# Test the system works
python webqa_pipeline.py \
  --reranker_model caption_lora \
  --generator_model noise_injected_lora \
  --filter 0 \
  --clip_topk 20
```

### Phase 2: Understanding & Analysis (Week 2-3)

#### 2.1 Study the Architecture

**Key Components to Understand:**
- **Retriever**: CLIP-based image-text embedding
- **Reranker**: MLLM-based relevance scoring
- **Generator**: Multi-modal answer generation
- **Training**: Two-stage process (in-batch → fixed retrieval)

#### 2.2 Run Ablation Studies

```python
# Example ablation commands
python webqa_pipeline.py --reranker_model caption_lora --generator_model vanilla  # w/o noise injection
python webqa_pipeline.py --reranker_model none --generator_model noise_injected_lora  # w/o reranker
```

#### 2.3 Code Analysis

**Study these key files:**
- `webqa_pipeline.py`: Main evaluation pipeline
- `finetune/scripts/`: Training scripts for different models
- `multimodal/`: Core multimodal processing logic

### Phase 3: Baseline Implementation (Week 3-4)

#### 3.1 Simple Multimodal RAG

```python
# src/simple_multimodal_rag.py
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss
import numpy as np
from PIL import Image

class SimpleMultimodalRAG:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # CLIP for retrieval
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # BLIP for generation
        self.generator_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        self.generator_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        
        self.image_index = None
        self.image_metadata = []
    
    def build_image_index(self, image_paths, captions=None):
        """Build FAISS index from images"""
        print(f"Building index from {len(image_paths)} images...")
        
        embeddings = []
        valid_metadata = []
        
        for i, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    # L2 normalize
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                embeddings.append(image_features.cpu().numpy())
                valid_metadata.append({
                    'path': img_path,
                    'caption': captions[i] if captions else None,
                    'index': i
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Build FAISS index
        embeddings = np.vstack(embeddings)
        self.image_index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product
        self.image_index.add(embeddings.astype('float32'))
        self.image_metadata = valid_metadata
        
        print(f"Built index with {len(valid_metadata)} images")
    
    def retrieve(self, query, k=5):
        """Retrieve top-k most relevant images"""
        if self.image_index is None:
            raise ValueError("Image index not built. Call build_image_index first.")
        
        # Encode query with CLIP
        inputs = self.clip_processor(text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Search in FAISS index
        scores, indices = self.image_index.search(text_features.cpu().numpy(), k)
        
        retrieved_items = []
        for idx, score in zip(indices[0], scores[0]):
            retrieved_items.append({
                'metadata': self.image_metadata[idx],
                'score': float(score)
            })
        
        return retrieved_items
    
    def generate(self, query, retrieved_images):
        """Generate answer using top retrieved image"""
        if not retrieved_images:
            return "No relevant images found."
        
        top_image_path = retrieved_images[0]['metadata']['path']
        image = Image.open(top_image_path).convert('RGB')
        
        # Generate with BLIP
        inputs = self.generator_processor(image, query, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.generator_model.generate(**inputs, max_length=50)
        
        answer = self.generator_processor.decode(generated_ids[0], skip_special_tokens=True)
        return answer
    
    def query(self, question, k=5):
        """End-to-end query processing"""
        retrieved = self.retrieve(question, k)
        answer = self.generate(question, retrieved)
        return {
            'answer': answer,
            'retrieved_images': retrieved,
            'query': question
        }
```

### Phase 4: Advanced Features (Week 4-6)

#### 4.1 Implement RagVL Features

**Key components to implement:**
- MLLM-based reranker
- Noise injection training
- Multi-stage training pipeline
- Adaptive filtering

#### 4.2 Training Pipeline

```python
# src/training/trainer.py
class MultimodalRAGTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    def train_stage1_in_batch(self, dataloader):
        """Stage 1: In-batch training"""
        self.model.train()
        for batch in dataloader:
            # Contrastive + Generative loss
            loss = self.compute_joint_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def train_stage2_fixed_retrieval(self, dataloader):
        """Stage 2: Fixed retrieval training"""
        # Pre-compute all embeddings
        self.build_global_index()
        
        self.model.train()
        for batch in dataloader:
            # Only generative loss with fixed retrievals
            loss = self.compute_generative_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
```

### Phase 5: Evaluation & Benchmarking (Week 6-7)

#### 5.1 Implement Evaluation Metrics

```python
# src/evaluation/metrics.py
def evaluate_webqa(predictions, references):
    """WebQA evaluation: BARTScore + Keyword Accuracy"""
    bart_scores = []
    keyword_accuracies = []
    
    for pred, ref in zip(predictions, references):
        # BARTScore for fluency
        bart_score = compute_bart_score(pred, ref)
        bart_scores.append(bart_score)
        
        # Keyword accuracy
        keyword_acc = compute_keyword_accuracy(pred, ref)
        keyword_accuracies.append(keyword_acc)
    
    return {
        'fluency': np.mean(bart_scores),
        'accuracy': np.mean(keyword_accuracies),
        'overall': np.mean(bart_scores) * np.mean(keyword_accuracies)
    }

def evaluate_multimodalqa(predictions, references):
    """MultimodalQA evaluation: Exact Match + F1"""
    em_scores = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        em = exact_match(pred, ref)
        f1 = f1_score(pred, ref)
        em_scores.append(em)
        f1_scores.append(f1)
    
    return {
        'exact_match': np.mean(em_scores),
        'f1': np.mean(f1_scores)
    }
```

### Phase 6: Novel Contributions (Week 7+)

#### 6.1 Research Extensions

**Potential improvements:**
1. **Better Reranking**: Use more sophisticated rerankers
2. **Cross-modal Attention**: Improve image-text fusion
3. **Domain Adaptation**: Specialize for specific domains
4. **Multi-hop Reasoning**: Handle complex questions requiring multiple retrieval steps
5. **Temporal Reasoning**: Handle time-sensitive information

#### 6.2 New Evaluation Scenarios

```python
# Create domain-specific benchmarks
# Example: Scientific paper QA
class ScientificPaperQA:
    def __init__(self):
        self.papers = []  # PDF papers with figures and text
        self.questions = []  # Questions requiring both text and figure understanding
    
    def evaluate_model(self, model):
        # Evaluate on scientific reasoning tasks
        pass
```

## 💡 Research Extensions & Ideas

### Architecture Improvements
- **Multi-scale Retrieval**: Retrieve at different granularities (patch, object, scene)
- **Progressive Curriculum**: Start with easier samples, gradually increase difficulty
- **Cross-modal Memory**: Maintain persistent cross-modal memory across queries
- **Adaptive Retrieval**: Dynamically determine how many items to retrieve

### Training Innovations
- **Hard Negative Mining**: Better strategies for selecting hard negatives
- **Contrastive Learning**: Improved contrastive objectives
- **Meta-learning**: Learn to adapt quickly to new domains
- **Federated Learning**: Train across multiple institutions without data sharing

### Applications
- **Scientific Literature**: Answer questions about research papers with figures/tables
- **Medical Imaging**: Clinical decision support with multimodal medical data
- **Educational Content**: Interactive textbook question answering
- **Technical Documentation**: Help with complex technical manuals

### Robustness & Reliability
- **Hallucination Detection**: Identify when the model is making things up
- **Confidence Estimation**: Provide uncertainty estimates
- **Bias Mitigation**: Address biases in multimodal representations
- **Adversarial Robustness**: Handle adversarial attacks on images/text

## 📚 Resources and References

### Key Papers

1. **MuRAG (2022)**: [arXiv:2210.02928](https://arxiv.org/abs/2210.02928)
2. **RagVL (2024)**: [arXiv:2407.21439](https://arxiv.org/abs/2407.21439)
3. **MMed-RAG (2024)**: Medical domain RAG
4. **RULE (2024)**: [EMNLP 2024] Reliable multimodal RAG
5. **Survey (2025)**: "Ask in Any Modality: A Comprehensive Survey on Multimodal RAG"

### Code Repositories

- **[IDEA-FinAI/RagVL](https://github.com/IDEA-FinAI/RagVL)**: State-of-the-art implementation
- **[richard-peng-xia/MMed-RAG](https://github.com/richard-peng-xia/MMed-RAG)**: Medical RAG
- **[llm-lab-org/Multimodal-RAG-Survey](https://github.com/llm-lab-org/Multimodal-RAG-Survey)**: Comprehensive survey
- **[Azure-Samples/multimodal-rag-code-execution](https://github.com/Azure-Samples/multimodal-rag-code-execution)**: Enterprise implementation

### Datasets

- **WebQA**: Multi-hop, multimodal question answering
- **MultimodalQA**: Questions over text, tables, and images  
- **VQA**: Visual Question Answering
- **OK-VQA**: Outside knowledge VQA
- **TextVQA**: Text-based VQA

### Tools & Libraries

- **Core**: PyTorch, Transformers, FAISS
- **Vision**: CLIP, BLIP, LLaVA, InternVL
- **Text**: T5, BERT, RoBERTa
- **Vector DB**: FAISS, Chroma, Pinecone
- **Evaluation**: BARTScore, ROUGE, BLEU

## 🎯 Success Metrics

### Technical Metrics
- **WebQA Overall Score**: Fluency × Accuracy
- **MultimodalQA**: Exact Match + F1
- **Inference Speed**: Time per query
- **Memory Usage**: RAM/VRAM requirements

### Research Metrics
- **Novel Contribution**: What's new compared to existing work?
- **Reproducibility**: Can others replicate your results?
- **Generalization**: Does it work across different domains?
- **Practical Impact**: Real-world applicability

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Try mixed precision training

2. **Slow Training**
   - Use DataParallel or DistributedDataParallel
   - Optimize data loading with multiple workers
   - Pre-compute embeddings when possible

3. **Poor Performance**
   - Check data preprocessing
   - Verify loss functions are correctly implemented
   - Monitor training curves for overfitting

### Debug Tools

```python
# Memory debugging
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Model debugging
def print_model_size(model):
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count/1e6:.2f}M")
```

## 📝 Project Timeline

### Week 1: Setup & Exploration
- [ ] Environment setup with uv/venv
- [ ] Clone and run RagVL
- [ ] Test basic functionality
- [ ] Literature review of key papers

### Week 2-3: Understanding
- [ ] Deep dive into RagVL architecture
- [ ] Run ablation studies
- [ ] Analyze failure cases
- [ ] Implement simple baseline

### Week 4-5: Implementation
- [ ] Build core components from scratch
- [ ] Implement training pipeline
- [ ] Add evaluation metrics
- [ ] Compare with baselines

### Week 6-7: Extensions
- [ ] Identify improvement opportunities
- [ ] Implement novel features
- [ ] Create new evaluation scenarios
- [ ] Write up results

### Week 8+: Research Contributions
- [ ] Refine novel contributions
- [ ] Comprehensive evaluation
- [ ] Write paper/report
- [ ] Open source implementation

---

## 🚀 Getting Started Checklist

- [ ] Choose implementation approach (RagVL recommended)
- [ ] Set up development environment with uv
- [ ] Clone repository and verify setup
- [ ] Download datasets (WebQA, MultimodalQA)
- [ ] Run baseline evaluation
- [ ] Study codebase and architecture
- [ ] Plan your novel contributions

**Next Steps**: Start with RagVL setup and get the baseline working. Once you have results, we can discuss specific extensions and novel contributions for your project.

Good luck with your multimodal RAG implementation! 🎯