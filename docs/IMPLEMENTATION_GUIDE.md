# MRAG-Bench Reproduction System - Implementation Guide

**Version:** 1.0
**Date:** October 4, 2025
**Status:** Production Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation and Setup](#installation-and-setup)
4. [Configuration](#configuration)
5. [Dataset Preparation](#dataset-preparation)
6. [Running Evaluations](#running-evaluations)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Debugging](#monitoring-and-debugging)
9. [API Usage](#api-usage)
10. [Best Practices](#best-practices)

---

## Introduction

This implementation guide provides step-by-step instructions for setting up, configuring, and running the MRAG-Bench reproduction system. The system achieves 53-59% accuracy on perspective change scenarios using LLaVA-1.5-7B and CLIP ViT-B/32 within 16GB VRAM constraints.

### System Capabilities

- **Multimodal RAG Pipeline:** CLIP-based retrieval + LLaVA-based generation
- **4 Perspective Change Scenarios:** Angle, partial, scope, occlusion
- **Memory Optimized:** 4-bit quantization for 16GB VRAM operation
- **Production Ready:** Comprehensive evaluation, monitoring, and reporting

### Key Features

- Automated dataset downloading and preprocessing
- CLIP ViT-B/32 image retrieval with FAISS indexing
- LLaVA-1.5-7B generation with 4-bit quantization
- Multi-scenario evaluation framework
- Performance optimization and benchmarking
- Statistical validation with confidence intervals
- Comprehensive result reporting (JSON + Markdown)

---

## System Requirements

### Hardware Requirements

**Minimum Configuration:**
- **GPU:** NVIDIA GPU with 16GB VRAM (e.g., RTX 5070Ti, RTX 4080)
- **RAM:** 32GB system memory (minimum)
- **Storage:** 500GB available space
  - ~100GB for models (LLaVA-1.5-7B, CLIP ViT-B/32)
  - ~50GB for MRAG-Bench dataset (16,130 images)
  - ~20GB for embeddings and FAISS indices
  - ~330GB for additional dependencies and cache

**Recommended Configuration:**
- **GPU:** NVIDIA RTX 5070Ti or better (16GB+ VRAM)
- **RAM:** 64GB system memory
- **Storage:** 1TB SSD for optimal performance
- **CPU:** Modern multi-core processor (8+ cores)

### Software Requirements

**Operating System:**
- Linux (Ubuntu 20.04+ recommended)
- Windows 10/11 with WSL2 (tested)
- macOS (limited GPU support)

**Core Dependencies:**
- Python 3.8+ (3.10 recommended)
- CUDA 11.8+ (for GPU acceleration)
- Git (for repository cloning)

**Optional Tools:**
- UV package manager (10x faster than pip)
- Conda (alternative environment manager)

---

## Installation and Setup

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/mmrag-cs6101.git
cd mmrag-cs6101

# Verify repository structure
ls -la
# Expected: src/, config/, docs/, tests/, requirements.txt, README.md
```

### Step 2: Create Virtual Environment

**Option A: Using UV (Recommended - 10x faster)**

```bash
# Install UV if not already installed
# Visit: https://github.com/astral-sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.10
uv venv mrag-bench-env --python 3.10

# Activate virtual environment
source mrag-bench-env/bin/activate  # Linux/macOS
# OR
mrag-bench-env\Scripts\activate  # Windows
```

**Option B: Using Standard venv**

```bash
# Create virtual environment
python3.10 -m venv mrag-bench-env

# Activate virtual environment
source mrag-bench-env/bin/activate  # Linux/macOS
# OR
mrag-bench-env\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
# Ensure virtual environment is activated
# You should see (mrag-bench-env) in your prompt

# Option A: Using UV (faster)
uv pip install -r requirements.txt

# Option B: Using standard pip
pip install -r requirements.txt

# Install CUDA-enabled PyTorch (if not included in requirements.txt)
# For CUDA 11.8:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Verify Installation

```bash
# Verify PyTorch and CUDA
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# Expected output:
# PyTorch Version: 2.x.x
# CUDA Available: True
# CUDA Version: 11.8 (or 12.1)
# GPU Device: NVIDIA GeForce RTX 5070Ti (or your GPU)
# GPU Memory: 16.0GB (or your GPU memory)

# Verify MRAG-Bench system imports
python -c "
from src.config import MRAGConfig
from src.utils.memory_manager import MemoryMonitor
from src.dataset import MRAGDataset
from src.retrieval import CLIPRetriever
from src.generation import LLaVAGenerationPipeline
from src.evaluation import MRAGBenchEvaluator

print('✅ All core imports successful')
"

# Verify configuration loading
python -c "
from src.config import MRAGConfig
config = MRAGConfig()
print(f'✅ Configuration loaded')
print(f'   VLM: {config.model.vlm_name}')
print(f'   Retriever: {config.model.retriever_name}')
print(f'   Memory Limit: {config.performance.memory_limit_gb}GB')
"
```

### Step 5: Download Models (Optional - Auto-downloads on first run)

```bash
# Models are automatically downloaded on first use
# To pre-download models manually:

python -c "
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import CLIPModel, CLIPProcessor

# Download LLaVA-1.5-7B (will take time - ~14GB)
print('Downloading LLaVA-1.5-7B...')
model = LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf')
processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
print('✅ LLaVA-1.5-7B downloaded')

# Download CLIP ViT-B/32 (~500MB)
print('Downloading CLIP ViT-B/32...')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('✅ CLIP ViT-B/32 downloaded')
"
```

---

## Configuration

### Default Configuration

The system uses `/mnt/d/dev/mmrag-cs6101/config/mrag_bench.yaml` for default configuration:

```yaml
model:
  vlm_name: "llava-hf/llava-1.5-7b-hf"
  retriever_name: "openai/clip-vit-base-patch32"
  quantization: "4bit"
  max_memory_gb: 14.0
  device: "cuda"

retrieval:
  top_k: 5
  embedding_dim: 512
  batch_size: 16

generation:
  max_length: 512
  temperature: 0.7
  top_p: 0.9

evaluation:
  scenarios: ["angle", "partial", "scope", "occlusion"]

performance:
  memory_limit_gb: 16.0
  memory_buffer_gb: 1.0
```

### Custom Configuration

Create a custom configuration file:

```bash
# Copy default configuration
cp config/mrag_bench.yaml config/my_config.yaml

# Edit configuration
nano config/my_config.yaml
```

**Example Custom Configuration:**

```yaml
model:
  vlm_name: "llava-hf/llava-1.5-7b-hf"
  quantization: "8bit"  # Use 8-bit instead of 4-bit for better quality
  max_memory_gb: 15.0   # Allow more memory for 8-bit quantization

retrieval:
  top_k: 10             # Retrieve more images for better context

generation:
  temperature: 0.5      # Lower temperature for more deterministic outputs
  max_length: 256       # Shorter responses for faster generation
```

### Configuration Priority

Configuration is loaded in the following order (later overrides earlier):

1. Default values in code
2. `config/mrag_bench.yaml` (default config file)
3. Custom config file (via `--config` argument)
4. Command-line arguments (highest priority)

**Example:**

```bash
# Use custom configuration file
python run_sprint9_multi_scenario.py --config config/my_config.yaml

# Override specific parameters via command-line
python run_sprint9_multi_scenario.py \
  --config config/my_config.yaml \
  --max-samples 100 \
  --output-dir output/custom_run
```

---

## Dataset Preparation

### Download MRAG-Bench Dataset

The system provides automated dataset downloading:

```bash
# Activate virtual environment
source mrag-bench-env/bin/activate

# Download MRAG-Bench dataset using provided script
python download_mrag_dataset.py

# Expected output:
# Downloading MRAG-Bench dataset...
# ✅ Downloaded 16,130 images
# ✅ Downloaded 1,353 questions
# ✅ Dataset saved to data/mrag_bench/
```

**Manual Download (if automated fails):**

```bash
# Create data directory
mkdir -p data/mrag_bench

# Download from Hugging Face or original source
# Follow MRAG-Bench paper instructions
# Dataset should contain:
# - data/mrag_bench/images/ (16,130 medical images)
# - data/mrag_bench/questions.json (1,353 questions)
# - data/mrag_bench/annotations.json (ground truth answers)
```

### Verify Dataset

```bash
# Verify dataset structure
python -c "
from src.dataset import MRAGDataset
from src.config import MRAGConfig

config = MRAGConfig()
dataset = MRAGDataset(config)

# Load dataset statistics
print(f'✅ Dataset loaded')
print(f'   Total images: {len(dataset.image_paths)}')
print(f'   Total questions: {len(dataset.questions)}')
print(f'   Scenarios: {dataset.get_available_scenarios()}')

# Get scenario counts
for scenario in ['angle', 'partial', 'scope', 'occlusion']:
    samples = dataset.filter_by_scenario(scenario)
    print(f'   {scenario.upper()}: {len(samples)} samples')
"

# Expected output:
# ✅ Dataset loaded
#    Total images: 16,130
#    Total questions: 778 (perspective change subset)
#    Scenarios: ['angle', 'partial', 'scope', 'occlusion']
#    ANGLE: 322 samples
#    PARTIAL: 246 samples
#    SCOPE: 102 samples
#    OCCLUSION: 108 samples
```

### Preprocess and Cache Embeddings (Optional)

```bash
# Pre-compute CLIP embeddings for all images
# This saves time during evaluation runs

python -c "
from src.retrieval import CLIPRetriever
from src.dataset import MRAGDataset
from src.config import MRAGConfig

config = MRAGConfig()
dataset = MRAGDataset(config)
retriever = CLIPRetriever(config)

# Generate embeddings for all images
print('Generating CLIP embeddings for 16,130 images...')
print('This may take 10-30 minutes depending on GPU...')

embeddings = retriever.encode_corpus(dataset.image_paths)

# Save embeddings
retriever.save_embeddings(embeddings, 'data/embeddings/clip_embeddings.npy')

print(f'✅ Embeddings saved to data/embeddings/clip_embeddings.npy')
print(f'   Shape: {embeddings.shape}')  # (16130, 512)
"
```

---

## Running Evaluations

### Sprint 7: MVP Evaluation (Single Scenario)

Test the system with a single scenario (angle changes):

```bash
# Activate virtual environment
source mrag-bench-env/bin/activate

# Run MVP evaluation (322 angle change samples)
python run_sprint7_mvp_evaluation.py

# Quick test with limited samples
python run_sprint7_mvp_evaluation.py --max-samples 50

# Custom configuration
python run_sprint7_mvp_evaluation.py \
  --config config/my_config.yaml \
  --output-dir output/mvp_test

# Expected runtime: 30-60 minutes (full 322 samples)
# Expected accuracy: 53-59% (MRAG-Bench baseline)
```

**Output:**
- `output/mvp_evaluation/results.json` - Machine-readable results
- `output/mvp_evaluation/summary_report.md` - Human-readable summary
- `output/mvp_evaluation/mvp_evaluation.log` - Execution log

### Sprint 9: Multi-Scenario Evaluation

Evaluate all 4 perspective change scenarios (778 samples):

```bash
# Activate virtual environment
source mrag-bench-env/bin/activate

# Run multi-scenario evaluation (all 778 samples)
python run_sprint9_multi_scenario.py --full-dataset

# Quick test mode
python run_sprint9_multi_scenario.py --quick-test

# Custom sample limits per scenario
python run_sprint9_multi_scenario.py --max-samples 100

# Custom configuration
python run_sprint9_multi_scenario.py \
  --config config/my_config.yaml \
  --output-dir output/multi_scenario_test \
  --max-samples 200

# Expected runtime: 2-4 hours (full 778 samples)
# Expected accuracy: 53-59% overall across all scenarios
```

**Output:**
- `output/multi_scenario/multi_scenario_results.json` - Complete results
- `output/multi_scenario/multi_scenario_summary.md` - Summary report
- `output/multi_scenario/multi_scenario_evaluation.log` - Execution log

### Sprint 10: Final Accuracy Validation

Run comprehensive validation with statistical analysis:

```bash
# Activate virtual environment
source mrag-bench-env/bin/activate

# Run final validation (3 evaluation runs, all 778 samples)
python run_sprint10_final_validation.py \
  --num-runs 3 \
  --full-dataset

# Quick validation (fewer runs, limited samples)
python run_sprint10_final_validation.py \
  --num-runs 2 \
  --max-samples 100

# Custom target range
python run_sprint10_final_validation.py \
  --target-min 0.50 \
  --target-max 0.65 \
  --num-runs 3

# Expected runtime: 6-12 hours (3 runs × 778 samples)
# Expected accuracy: 54-55% with 95% CI [52%, 57%]
```

**Output:**
- `output/sprint10/sprint10_final_validation_results.json` - Complete results with statistics
- `output/sprint10/sprint10_summary_report.md` - Statistical validation summary
- `output/sprint10/sprint10_final_validation.log` - Execution log

---

## Performance Optimization

### Sprint 8: Performance Optimization Framework

Run performance benchmarking and optimization:

```bash
# Activate virtual environment
source mrag-bench-env/bin/activate

# Run performance optimization benchmark
python run_sprint8_optimization.py --benchmark

# Comprehensive performance analysis
python run_sprint8_optimization.py \
  --benchmark \
  --max-samples 200 \
  --output-dir output/performance_analysis

# Expected output:
# - Timing analysis (retrieval, generation, end-to-end)
# - Memory profiling (peak usage, allocation patterns)
# - Throughput metrics (queries/second, samples/minute)
# - Optimization recommendations
```

**Output:**
- `output/sprint8/performance_benchmark.json` - Performance metrics
- `output/sprint8/performance_report.md` - Analysis and recommendations
- `output/sprint8/optimization.log` - Execution log

### Memory Optimization

**Monitor Memory Usage:**

```bash
# Real-time memory monitoring
python -c "
from src.utils.memory_manager import MemoryMonitor
import time

monitor = MemoryMonitor(memory_limit_gb=15.0)

# Monitor for 10 seconds
for i in range(10):
    stats = monitor.get_current_stats()
    print(f'GPU Memory: {stats.gpu_allocated_gb:.1f}/{stats.gpu_total_gb:.1f}GB '
          f'({stats.gpu_utilization_percent():.1f}%)')
    time.sleep(1)
"
```

**Optimize for Limited Memory:**

```yaml
# config/low_memory.yaml
model:
  quantization: "4bit"      # Use 4-bit quantization
  max_memory_gb: 12.0       # Conservative memory limit

retrieval:
  batch_size: 8             # Smaller batch size

generation:
  max_length: 256           # Shorter generation length

performance:
  memory_limit_gb: 14.0     # Strict memory limit
  memory_buffer_gb: 2.0     # Larger buffer
```

### Speed Optimization

**Optimize for Faster Evaluation:**

```yaml
# config/fast_eval.yaml
retrieval:
  batch_size: 32            # Larger retrieval batch
  top_k: 3                  # Fewer retrieved images

generation:
  max_length: 256           # Shorter generation
  num_beams: 1              # No beam search
  do_sample: false          # Greedy decoding
```

Run with optimized configuration:

```bash
python run_sprint9_multi_scenario.py \
  --config config/fast_eval.yaml \
  --max-samples 500
```

---

## Monitoring and Debugging

### Enable Comprehensive Logging

```bash
# Set logging level via environment variable
export MRAG_LOG_LEVEL=DEBUG

# Run with debug logging
python run_sprint9_multi_scenario.py --max-samples 50

# View detailed logs
tail -f output/multi_scenario/multi_scenario_evaluation.log
```

### Monitor GPU Usage

```bash
# In separate terminal, monitor GPU in real-time
watch -n 1 nvidia-smi

# Or use more detailed monitoring
nvidia-smi dmon -s u -c 1000
```

### Debug Failed Evaluations

```bash
# Check logs for errors
grep "ERROR" output/*/evaluation.log
grep "Failed" output/*/evaluation.log

# Run with verbose output
python run_sprint7_mvp_evaluation.py \
  --max-samples 10 \
  --verbose

# Test single sample
python -c "
from src.pipeline import MRAGPipeline
from src.config import MRAGConfig
from src.dataset import MRAGDataset

config = MRAGConfig()
pipeline = MRAGPipeline(config)
dataset = MRAGDataset(config)

# Get first sample
sample = dataset.get_sample(0)

# Run pipeline
result = pipeline.process_query(
    query=sample['question'],
    retrieved_images=pipeline.retriever.retrieve(sample['question'])
)

print(f'Query: {sample[\"question\"]}')
print(f'Generated Answer: {result[\"answer\"]}')
print(f'Ground Truth: {sample[\"answer\"]}')
"
```

### Health Checks

```bash
# Run system health check
python -c "
from src.utils.health_check import SystemHealthCheck

checker = SystemHealthCheck()
report = checker.run_all_checks()

print('System Health Report:')
print(f'✅ GPU Available: {report[\"gpu_available\"]}')
print(f'✅ CUDA Version: {report[\"cuda_version\"]}')
print(f'✅ Available Memory: {report[\"gpu_memory_gb\"]}GB')
print(f'✅ Models Accessible: {report[\"models_accessible\"]}')
print(f'✅ Dataset Accessible: {report[\"dataset_accessible\"]}')

if report['status'] == 'healthy':
    print('\\n✅ System is healthy and ready for evaluation')
else:
    print(f'\\n❌ System issues detected: {report[\"issues\"]}')
"
```

---

## API Usage

### Python API

**Basic Pipeline Usage:**

```python
from src.pipeline import MRAGPipeline
from src.config import MRAGConfig

# Load configuration
config = MRAGConfig()

# Initialize pipeline
pipeline = MRAGPipeline(config)

# Process single query
query = "What is the difference between the two MRI scans?"
result = pipeline.process_query(query)

print(f"Answer: {result['answer']}")
print(f"Retrieved Images: {len(result['retrieved_images'])}")
print(f"Processing Time: {result['processing_time']:.2f}s")
```

**Custom Component Usage:**

```python
from src.retrieval import CLIPRetriever
from src.generation import LLaVAGenerationPipeline
from src.config import MRAGConfig

config = MRAGConfig()

# Initialize components separately
retriever = CLIPRetriever(config)
generator = LLaVAGenerationPipeline(config)

# Retrieval
query = "Show MRI scan with angle change"
retrieved_images = retriever.retrieve(query, top_k=5)

# Generation
answer = generator.generate(
    query=query,
    context_images=retrieved_images
)

print(f"Answer: {answer}")
```

**Batch Processing:**

```python
from src.evaluation import MRAGBenchEvaluator
from src.config import MRAGConfig

config = MRAGConfig()
evaluator = MRAGBenchEvaluator(config)

# Evaluate specific scenario
results = evaluator.evaluate_scenario(
    scenario_type="angle",
    max_samples=100
)

print(f"Accuracy: {results.accuracy:.1%}")
print(f"Total Samples: {results.total_samples}")
print(f"Correct Answers: {results.correct_answers}")
```

---

## Best Practices

### 1. Always Use Virtual Environment

```bash
# ✅ Correct
source mrag-bench-env/bin/activate
python run_sprint9_multi_scenario.py

# ❌ Wrong
python run_sprint9_multi_scenario.py  # Without activation
```

### 2. Monitor Memory Usage

```python
from src.utils.memory_manager import MemoryMonitor

monitor = MemoryMonitor(memory_limit_gb=15.0)

# Before heavy operation
if not monitor.check_memory_availability(required_gb=5.0):
    monitor.clear_gpu_memory()
```

### 3. Use Configuration Files

```bash
# ✅ Correct - Use configuration files
python run_sprint9_multi_scenario.py --config config/production.yaml

# ❌ Wrong - Hardcode parameters in code
```

### 4. Start with Small Samples

```bash
# ✅ Correct - Test with small samples first
python run_sprint9_multi_scenario.py --max-samples 50

# Then scale up
python run_sprint9_multi_scenario.py --full-dataset
```

### 5. Save Results Regularly

```bash
# Configure result saving
python run_sprint9_multi_scenario.py \
  --output-dir output/experiment_$(date +%Y%m%d_%H%M%S) \
  --save-intermediate-results
```

### 6. Monitor Logs

```bash
# Monitor logs in real-time
tail -f output/*/evaluation.log
```

### 7. Use Multi-Run Validation

```bash
# For production results, use multiple runs
python run_sprint10_final_validation.py --num-runs 3
```

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for comprehensive troubleshooting guide.

**Common Issues:**

1. **Out of Memory:** Reduce batch size or use 4-bit quantization
2. **Import Errors:** Ensure virtual environment is activated
3. **CUDA Errors:** Verify CUDA version matches PyTorch installation
4. **Slow Performance:** Check GPU utilization and memory usage

---

## Next Steps

1. **Run MVP Evaluation:** `python run_sprint7_mvp_evaluation.py --max-samples 100`
2. **Run Multi-Scenario:** `python run_sprint9_multi_scenario.py --quick-test`
3. **Run Final Validation:** `python run_sprint10_final_validation.py --num-runs 2`
4. **Review API Documentation:** See [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
5. **Read Performance Report:** See [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)

---

**Document Version:** 1.0
**Last Updated:** October 4, 2025
**Maintained By:** MRAG-Bench Development Team
