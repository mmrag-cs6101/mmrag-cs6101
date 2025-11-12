# MRAG-Bench Reproduction System - Troubleshooting Guide

**Version:** 1.0
**Date:** October 4, 2025
**Status:** Production Ready

---

## Table of Contents

1. [Common Issues and Solutions](#common-issues-and-solutions)
2. [Environment Issues](#environment-issues)
3. [CUDA and GPU Issues](#cuda-and-gpu-issues)
4. [Memory Issues](#memory-issues)
5. [Model Loading Issues](#model-loading-issues)
6. [Dataset Issues](#dataset-issues)
7. [Performance Issues](#performance-issues)
8. [Evaluation Issues](#evaluation-issues)
9. [Diagnostic Tools](#diagnostic-tools)

---

## Common Issues and Solutions

### Issue: ModuleNotFoundError

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Cause:** Virtual environment not activated or dependencies not installed.

**Solution:**
```bash
# Activate virtual environment
source mrag-bench-env/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Verify installation
python -c "import src; print('✅ Success')"
```

---

### Issue: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Cause:** Model or batch size exceeds 16GB VRAM.

**Solutions:**

**1. Use 4-bit Quantization (Recommended)**
```yaml
# config/mrag_bench.yaml
model:
  quantization: "4bit"  # Change from "8bit" or "none"
  max_memory_gb: 12.0   # Reduce if needed
```

**2. Reduce Batch Sizes**
```yaml
dataset:
  batch_size: 2  # Reduce from 4

retrieval:
  batch_size: 8  # Reduce from 16
```

**3. Reduce Generation Length**
```yaml
generation:
  max_length: 256  # Reduce from 512
```

**4. Manual Memory Cleanup**
```python
from src.utils.memory_manager import MemoryManager

manager = MemoryManager()
manager.clear_gpu_memory()
```

---

### Issue: Slow Performance

**Symptom:** Processing time >60s per query.

**Causes and Solutions:**

**1. GPU Not Being Used**
```bash
# Check GPU utilization
nvidia-smi

# If 0% utilization, verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**2. Batch Size Too Small**
```yaml
retrieval:
  batch_size: 32  # Increase for faster embedding generation
```

**3. FAISS Index Not Cached**
```bash
# Ensure index is built and saved
ls -lh data/embeddings/faiss_index.bin
```

---

## Environment Issues

### Virtual Environment Not Activated

**Symptom:** Import errors or wrong Python version.

**Check:**
```bash
which python
# Should show: /path/to/mrag-bench-env/bin/python
```

**Solution:**
```bash
source mrag-bench-env/bin/activate
```

---

### Wrong Python Version

**Symptom:** Syntax errors or incompatible features.

**Check:**
```bash
python --version
# Should be: Python 3.8+ (3.10 recommended)
```

**Solution:**
```bash
# Recreate environment with correct Python
rm -rf mrag-bench-env
uv venv mrag-bench-env --python 3.10
source mrag-bench-env/bin/activate
uv pip install -r requirements.txt
```

---

### Missing Dependencies

**Symptom:** ImportError for specific packages.

**Solution:**
```bash
source mrag-bench-env/bin/activate

# Reinstall all dependencies
uv pip install -r requirements.txt

# Install specific missing package
uv pip install <package-name>
```

---

## CUDA and GPU Issues

### CUDA Not Available

**Symptom:**
```python
torch.cuda.is_available()  # Returns False
```

**Diagnostic:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

**Solutions:**

**1. Reinstall PyTorch with CUDA**
```bash
# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**2. Update NVIDIA Driver**
```bash
# Ubuntu/Debian
sudo ubuntu-drivers autoinstall

# Check driver version
nvidia-smi
```

---

### CUDA Version Mismatch

**Symptom:** CUDA errors during model loading.

**Check Versions:**
```bash
# System CUDA
nvcc --version

# PyTorch CUDA
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

**Solution:** Install PyTorch matching your CUDA version (see above).

---

### Multiple GPUs Detected

**Symptom:** Model loads on wrong GPU.

**Solution:**
```bash
# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# Or in Python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

---

## Memory Issues

### GPU Memory Leak

**Symptom:** Memory usage increases with each query.

**Diagnostic:**
```python
from src.utils.memory_manager import MemoryManager

manager = MemoryManager()

# Before query
before = manager.get_memory_stats()
print(f"Before: {before.gpu_allocated_gb:.2f}GB")

# Process query
result = pipeline.process_query("test")

# After query
after = manager.get_memory_stats()
print(f"After: {after.gpu_allocated_gb:.2f}GB")
print(f"Leaked: {after.gpu_allocated_gb - before.gpu_allocated_gb:.2f}GB")
```

**Solution:**
```python
# Force garbage collection after each query
import gc
import torch

result = pipeline.process_query("test")
gc.collect()
torch.cuda.empty_cache()
```

---

### System RAM Exhausted

**Symptom:** System freezing or swap usage.

**Solutions:**

**1. Reduce Dataset Loading**
```python
# Load dataset in streaming mode
dataset = MRAGDataset(batch_size=2)  # Smaller batches
```

**2. Limit Concurrent Processes**
```bash
# Run evaluation with limited parallelism
python run_sprint9_multi_scenario.py --max-samples 50
```

---

## Model Loading Issues

### Model Download Fails

**Symptom:** Hugging Face download timeouts or errors.

**Solutions:**

**1. Set Hugging Face Token (for gated models)**
```bash
export HF_TOKEN="your_token_here"
```

**2. Use Offline Mode (if models already downloaded)**
```bash
export TRANSFORMERS_OFFLINE=1
```

**3. Manual Download**
```python
from transformers import LlavaNextForConditionalGeneration

# Download with retries
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    resume_download=True
)
```

---

### Quantization Errors

**Symptom:**
```
bitsandbytes not compiled with CUDA support
```

**Solution:**
```bash
# Reinstall bitsandbytes
uv pip uninstall bitsandbytes
uv pip install bitsandbytes>=0.39.0

# Verify CUDA support
python -c "import bitsandbytes as bnb; print(bnb.cuda_setup.common.get_compute_capability())"
```

---

### Model Weights Mismatch

**Symptom:** Unexpected model behavior or errors.

**Solution:**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/transformers/*

# Re-download models
python -c "
from transformers import LlavaNextForConditionalGeneration
model = LlavaNextForConditionalGeneration.from_pretrained(
    'llava-hf/llava-1.5-7b-hf',
    force_download=True
)
"
```

---

## Dataset Issues

### Dataset Not Found

**Symptom:**
```
FileNotFoundError: data/mrag_bench not found
```

**Solution:**
```bash
# Download dataset
python download_mrag_dataset.py

# Verify structure
ls -R data/mrag_bench/
# Should have: images/, questions.json, annotations.json
```

---

### Missing Images

**Symptom:** Some image paths return FileNotFoundError.

**Diagnostic:**
```python
from src.dataset import MRAGDataset

dataset = MRAGDataset()
validation = dataset.validate_dataset()

if validation["status"] == "error":
    print(f"Missing images: {validation['missing_images']}")
```

**Solution:**
```bash
# Re-download dataset
python download_mrag_dataset.py --force-download
```

---

### Corrupted Images

**Symptom:** PIL image loading errors.

**Diagnostic:**
```python
from PIL import Image
import os

data_dir = "data/mrag_bench/images"
for filename in os.listdir(data_dir):
    try:
        img = Image.open(os.path.join(data_dir, filename))
        img.verify()
    except Exception as e:
        print(f"Corrupted: {filename} - {e}")
```

**Solution:** Re-download corrupted images or entire dataset.

---

## Performance Issues

### Slow Image Encoding

**Symptom:** CLIP encoding takes >10s for 16 images.

**Solutions:**

**1. Increase Batch Size**
```yaml
retrieval:
  batch_size: 32  # Increase from 16
```

**2. Use Mixed Precision**
```python
# CLIP automatically uses float16 on CUDA
# Verify:
print(f"CLIP dtype: {retriever.model.dtype}")
```

**3. Pre-compute Embeddings**
```bash
# Generate embeddings once
python -c "
from src.retrieval import CLIPRetriever
from src.dataset import MRAGDataset
from src.config import MRAGConfig

config = MRAGConfig()
dataset = MRAGDataset(config)
retriever = CLIPRetriever(config.retrieval)

corpus = dataset.get_retrieval_corpus()
embeddings = retriever.encode_corpus(corpus)
retriever.save_embeddings(embeddings)
"
```

---

### Slow Generation

**Symptom:** LLaVA generation >40s per query.

**Solutions:**

**1. Reduce Max Length**
```yaml
generation:
  max_length: 256  # Reduce from 512
```

**2. Disable Sampling**
```yaml
generation:
  do_sample: false  # Use greedy decoding
  temperature: 1.0
```

**3. Verify Quantization**
```python
print(f"Model dtype: {generator.model.dtype}")
# Should show: torch.uint8 (for 4-bit)
```

---

### FAISS Index Slow

**Symptom:** Similarity search >5s.

**Solutions:**

**1. Rebuild Index with GPU**
```python
retriever.build_index(corpus_paths, use_gpu=True)
```

**2. Optimize Index Parameters**
```yaml
retrieval:
  faiss_index_type: "Flat"  # Use exact search for smaller corpuses
```

---

## Evaluation Issues

### Evaluation Fails Midway

**Symptom:** Evaluation stops after N samples.

**Diagnostic:**
```bash
# Check logs
tail -100 output/*/evaluation.log | grep -i error
```

**Solutions:**

**1. Enable Auto-Recovery**
```python
evaluator = MRAGBenchEvaluator(config)
evaluator.enable_auto_recovery = True  # Retry failed samples
```

**2. Reduce Sample Size**
```bash
python run_sprint9_multi_scenario.py --max-samples 50
```

**3. Check Memory**
```python
manager = MemoryManager()
stats = manager.get_memory_stats()
if not stats.memory_available:
    manager.emergency_cleanup()
```

---

### Incorrect Accuracy

**Symptom:** Accuracy significantly different from expected (53-59%).

**Diagnostic Checklist:**

1. **Verify Dataset**
   ```python
   dataset.validate_dataset()
   ```

2. **Check Model Versions**
   ```python
   print(f"LLaVA: {generator.config.model_name}")
   print(f"CLIP: {retriever.config.model_name}")
   ```

3. **Verify Configuration**
   ```python
   config = MRAGConfig.from_yaml("config/mrag_bench.yaml")
   print(config.to_dict())
   ```

4. **Sample Evaluation**
   ```python
   # Test with known samples
   sample = dataset.get_sample(0)
   result = pipeline.process_query(sample.question)
   print(f"Expected: {sample.answer}")
   print(f"Generated: {result.generated_answer}")
   ```

---

## Diagnostic Tools

### System Health Check

```python
from src.utils.health_check import SystemHealthCheck

checker = SystemHealthCheck()
report = checker.run_all_checks()

print("System Health Report:")
for check, status in report.items():
    print(f"  {check}: {'✅' if status else '❌'}")
```

### Memory Profiler

```python
from src.utils.memory_manager import MemoryManager
import time

manager = MemoryManager()

# Profile operation
manager.start_profiling("model_loading")
# ... operation ...
profile = manager.end_profiling("model_loading")

print(f"Peak memory: {profile.peak_memory_gb:.2f}GB")
print(f"Duration: {profile.duration:.2f}s")
```

### Performance Benchmark

```bash
# Run comprehensive benchmark
python run_sprint8_optimization.py --benchmark --max-samples 100

# Results in: output/sprint8/performance_benchmark.json
```

### Log Analysis

```bash
# Find errors
grep -i "error" output/*/evaluation.log

# Find warnings
grep -i "warning" output/*/evaluation.log

# Find memory issues
grep -i "memory" output/*/evaluation.log

# Find slow queries
grep "processing_time.*[5-9][0-9]\." output/*/evaluation.log
```

---

## Getting Help

If issues persist after trying these solutions:

1. **Check System Requirements**
   - GPU: 16GB VRAM minimum
   - RAM: 32GB minimum
   - CUDA: 11.8+ or 12.1+

2. **Verify Installation**
   ```bash
   python -c "
   import torch
   from src.config import MRAGConfig
   print(f'✅ PyTorch: {torch.__version__}')
   print(f'✅ CUDA: {torch.cuda.is_available()}')
   print(f'✅ MRAG-Bench: OK')
   "
   ```

3. **Collect Diagnostics**
   ```bash
   # Save system info
   python -c "
   import torch
   import sys
   print(f'Python: {sys.version}')
   print(f'PyTorch: {torch.__version__}')
   print(f'CUDA: {torch.version.cuda}')
   print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
   " > diagnostics.txt

   # Attach diagnostics.txt when seeking help
   ```

4. **Review Documentation**
   - [Implementation Guide](IMPLEMENTATION_GUIDE.md)
   - [API Documentation](API_DOCUMENTATION.md)
   - [Performance Analysis](PERFORMANCE_ANALYSIS.md)

---

**Document Version:** 1.0
**Last Updated:** October 4, 2025
**Maintained By:** MRAG-Bench Development Team
