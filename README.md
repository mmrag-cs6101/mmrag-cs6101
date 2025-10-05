# MRAG-Bench Reproduction System

A modular multimodal retrieval-augmented generation system for reproducing MRAG-Bench baseline results on perspective change scenarios using multiple-choice evaluation.

## 🎯 Project Overview

This system implements multimodal RAG evaluation on the official MRAG-Bench dataset, focusing on:
- **LLaVA-1.5-7B** for vision-language understanding
- **CLIP ViT-B/32** for image-text retrieval
- **4-bit quantization** for memory efficiency
- **16GB VRAM** constraint optimization
- **Multiple-choice evaluation** (A/B/C/D format)
- **Target: 53-59% accuracy** on perspective change scenarios

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB VRAM recommended)
- UV package manager (recommended - 10x faster than pip)

### Environment Setup

**⚠️ Important: Always use the virtual environment for all operations**

```bash
# 1. Clone repository
git clone <your-repo-url>
cd mmrag-cs6101

# 2. Create virtual environment
uv venv mrag-bench-env --python 3.10

# 3. Activate virtual environment (REQUIRED for all commands)
source mrag-bench-env/bin/activate  # Linux/macOS
# mrag-bench-env\Scripts\activate   # Windows

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Install PyTorch with CUDA support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
# Ensure virtual environment is activated
source mrag-bench-env/bin/activate

# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test MRAG-Bench imports
python -c "from src.config import MRAGConfig; print('✅ MRAG-Bench system ready')"
```

## 🧪 Running Evaluations

The system uses the official HuggingFace dataset (`uclanlp/MRAG-Bench`) with multiple-choice questions.

### Quick Test (40 samples, ~5 minutes)

```bash
# Activate environment
source mrag-bench-env/bin/activate

# Run quick evaluation
python eval_simple.py
```

**Expected output:**
```
[1/40] Scenario: Scope | Predicted: B | Correct: A | ✗
[2/40] Scenario: Obstruction | Predicted: C | Correct: C | ✓
...
EVALUATION RESULTS (First 40 samples)
Accuracy: 45.0% (18/40)
Target Range: 53-59%
Status: ✗ FAIL
```

### Full Evaluation (1353 samples, ~4-6 hours)

```bash
# Activate environment
source mrag-bench-env/bin/activate

# Run full evaluation on entire dataset
python eval_full.py
```

**Features:**
- Progress updates every 50 samples with ETA
- Per-scenario accuracy breakdown (angle, partial, scope, occlusion)
- Detailed results saved to `output/full_evaluation/` with timestamp
- Overall and per-scenario statistics

**Expected output:**
```
Progress: [50/1353] | Current Accuracy: 46.0% | ETA: 238.5m
Progress: [100/1353] | Current Accuracy: 45.5% | ETA: 225.3m
...
FULL MRAG-BENCH EVALUATION RESULTS
Overall Accuracy: 45.2% (612/1353)
Target Range: 53-59%
Status: ✗ FAIL
Evaluation Time: 285.3 minutes

Per-Scenario Results:
  angle       :  42.5% (180/424)
  partial     :  48.2% (145/301)
  scope       :  44.8% (245/547)
  occlusion   :  51.2% (42/82)

Results saved to: output/full_evaluation/full_evaluation_20251005_001234.json
```

## 📊 Dataset Information

### Official MRAG-Bench Format

The system uses `uclanlp/MRAG-Bench` from HuggingFace:
- **1,353 questions** across 9 scenarios
- **Multiple-choice format** (A/B/C/D)
- **14,475 images** total in corpus
- **5 ground-truth images** per question
- **5 retrieved images** per question (pre-computed)

### Question Example

```
Question: Can you identify this animal?
A. silky_terrier  ← Correct answer
B. Yorkshire_terrier
C. Australian_terrier
D. Cairn_terrier
```

### Dataset Loading

The dataset is automatically downloaded from HuggingFace on first run:

```bash
source mrag-bench-env/bin/activate
python -c "from datasets import load_dataset; ds = load_dataset('uclanlp/MRAG-Bench', split='test'); print(f'Loaded {len(ds)} samples')"
```

## 🏗️ System Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│              MRAG-Bench Evaluation Pipeline                  │
└─────────────────────────────────────────────────────────────┘

1. Load Question + Ground-Truth Images
   ↓
2. Present Multiple-Choice Question to LLaVA
   - Question: "Can you identify this animal?"
   - Images: 3 ground-truth images (of 5 available)
   - Choices: A/B/C/D options
   ↓
3. LLaVA Generation (~1-2s)
   - Process images + question through LLaVA-1.5-7B (4-bit)
   - Generate answer choice: "A", "B", "C", or "D"
   ↓
4. Extract Answer Choice & Evaluate
   - Parse model output for letter (A/B/C/D)
   - Compare to ground truth
   - Track accuracy
```

### Current Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | ~45% | 18/40 on test samples |
| **Target** | 53-59% | MRAG-Bench baseline |
| **GPU Memory** | ~4GB | LLaVA-1.5-7B with 4-bit quantization |
| **Generation Time** | 0.5-2s | Per question |
| **Images Used** | 3 of 5 | Ground-truth images per question |

### Accuracy Timeline

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Text-to-image retrieval (open-ended) | 2.5% | Incorrect architecture |
| Image-to-image retrieval (open-ended) | 5.0% | Still wrong format |
| **Ground-truth images (multiple-choice)** | **45.0%** | ✅ Current approach |
| Target | 53-59% | 8-14% gap remaining |

## 📁 Project Structure

```
mmrag-cs6101/
├── mrag-bench-env/              # 🔴 Virtual environment (REQUIRED)
├── src/
│   ├── dataset/
│   │   ├── mrag_hf_dataset.py   # HuggingFace dataset wrapper
│   │   └── mrag_dataset.py      # Legacy local dataset (deprecated)
│   ├── retrieval/
│   │   └── clip_retriever.py    # CLIP-based retrieval (optional)
│   ├── generation/
│   │   ├── llava_pipeline.py    # LLaVA generation with MC prompts
│   │   └── interface.py         # Generation interfaces
│   ├── evaluation/
│   │   └── evaluator.py         # Evaluation framework
│   ├── utils/
│   │   └── memory_manager.py    # GPU memory management
│   └── config.py                # Configuration
├── config/
│   └── mrag_bench.yaml          # System configuration
├── eval_simple.py               # Quick test (40 samples)
├── eval_full.py                 # Full evaluation (1353 samples)
├── output/
│   └── full_evaluation/         # Evaluation results (JSON)
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Hardware Requirements

- **GPU**: 16GB VRAM (RTX 5070Ti, RTX 4090, etc.)
- **RAM**: 32GB system RAM recommended
- **Storage**: ~20GB for models and dataset

### Configuration File

Located at `config/mrag_bench.yaml`:

```yaml
model:
  vlm_name: "llava-hf/llava-1.5-7b-hf"
  retriever_name: "openai/clip-vit-base-patch32"
  quantization: "4bit"
  device: "cuda"

generation:
  max_length: 10              # Short answers (A/B/C/D)
  temperature: 0.1            # Low for deterministic output
  do_sample: false            # Greedy decoding

performance:
  memory_limit_gb: 16.0
```

## 🧪 Testing

```bash
# Activate environment
source mrag-bench-env/bin/activate

# Run test suite
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_config.py -v
python -m pytest tests/test_memory_manager.py -v
```

## 📈 Results Analysis

### Viewing Results

```bash
# List all evaluation results
ls -lh output/full_evaluation/

# View latest results
cat output/full_evaluation/full_evaluation_*.json | python -m json.tool | head -50

# Extract accuracy
python -c "
import json
from pathlib import Path
results = sorted(Path('output/full_evaluation').glob('*.json'))[-1]
with open(results) as f:
    data = json.load(f)
print(f'Overall Accuracy: {data[\"overall_accuracy\"]*100:.1f}%')
for scenario, stats in data['scenario_results'].items():
    print(f'{scenario:12s}: {stats[\"accuracy\"]*100:5.1f}% ({stats[\"correct\"]}/{stats[\"total\"]})')
"
```

### Results File Format

```json
{
  "timestamp": "2025-10-05T00:12:34",
  "overall_accuracy": 0.452,
  "total_samples": 1353,
  "correct_answers": 612,
  "target_range": [0.53, 0.59],
  "target_achieved": false,
  "scenario_results": {
    "angle": {"accuracy": 0.425, "correct": 180, "total": 424},
    "partial": {"accuracy": 0.482, "correct": 145, "total": 301},
    "scope": {"accuracy": 0.448, "correct": 245, "total": 547},
    "occlusion": {"accuracy": 0.512, "correct": 42, "total": 82}
  },
  "configuration": {
    "model": "llava-hf/llava-1.5-7b-hf",
    "num_gt_images_used": 3,
    "temperature": 0.1
  }
}
```

## 🎯 Improving Accuracy

Current accuracy is ~45%, target is 53-59%. Potential improvements:

1. **Use all 5 GT images instead of 3**
   - Currently using first 3 images
   - Full 5 images may provide better context

2. **Prompt engineering**
   - Optimize multiple-choice prompt format
   - Add examples or reasoning chains

3. **Temperature tuning**
   - Current: 0.1
   - MRAG-Bench paper uses 0.0 (completely deterministic)

4. **Model size**
   - Current: LLaVA-1.5-7B
   - Try LLaVA-1.5-13B or LLaVA-OneVision

## ⚠️ Important Notes

1. **Always use virtual environment** - All Python operations must run within `mrag-bench-env`
2. **First run downloads models** - ~14GB download for LLaVA (cached after first run)
3. **Dataset auto-downloads** - HuggingFace dataset downloads automatically on first use
4. **GPU memory management** - System optimized for 16GB VRAM with 4-bit quantization
5. **Multiple-choice format** - MRAG-Bench is MC (A/B/C/D), not open-ended generation

## 🆘 Troubleshooting

### Virtual Environment Issues

```bash
# Error: ModuleNotFoundError
# ✅ Solution: Always activate environment first
source mrag-bench-env/bin/activate
python eval_simple.py
```

### CUDA/GPU Issues

```bash
# Check GPU availability
source mrag-bench-env/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Dataset Download Issues

```bash
# Force re-download dataset
source mrag-bench-env/bin/activate
python -c "from datasets import load_dataset; load_dataset('uclanlp/MRAG-Bench', split='test', download_mode='force_redownload')"
```

### Out of Memory

```bash
# Reduce batch size or images used
# Edit eval_simple.py or eval_full.py:
# images = sample["gt_images"][:3]  # Reduce from 3 to 2 or 1
```

## 📖 Additional Documentation

- **[CLAUDE.md](CLAUDE.md)** - Detailed development guide for Claude Code
- **Configuration**: `config/mrag_bench.yaml`
- **Sprint Progress**: See git commit history for development timeline

## 🤝 Contributing

1. **Always use virtual environment**:
   ```bash
   source mrag-bench-env/bin/activate
   ```

2. **Run tests before committing**:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Follow the evaluation format**:
   - Use official HuggingFace dataset
   - Multiple-choice format (A/B/C/D)
   - Ground-truth images

---

**⚠️ Remember: Virtual environment activation is required for ALL operations!**

```bash
# Always start with this
source mrag-bench-env/bin/activate
```
