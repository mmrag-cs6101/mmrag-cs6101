# MRAG-Bench Reproduction System

A modular multimodal retrieval-augmented generation system for reproducing MRAG-Bench baseline results on perspective change scenarios.

## 🎯 Project Overview

This system implements a fresh approach to multimodal RAG, focusing on reproducing the MRAG-Bench paper results with:
- **LLaVA-1.5-7B** + **CLIP ViT-B/32** architecture
- **4-bit quantization** for memory efficiency
- **16GB VRAM** constraint optimization
- **53-59% accuracy target** on perspective change scenarios

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (16GB VRAM recommended)
- UV package manager (10x faster than pip)

### Environment Setup (Required)

**⚠️ Important: Always use the virtual environment for all operations**

#### Option 1: Using UV (Recommended - 10x faster)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd mmrag-cs6101

# 2. Create virtual environment
uv venv mrag-bench-env --python 3.10

# 3. Activate virtual environment
source mrag-bench-env/bin/activate  # Linux/macOS
# mrag-bench-env\Scripts\activate   # Windows

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Install CUDA PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Option 2: Using Standard Pip

```bash
# 1. Clone repository
git clone <your-repo-url>
cd mmrag-cs6101

# 2. Create virtual environment
python -m venv mrag-bench-env

# 3. Activate virtual environment
source mrag-bench-env/bin/activate  # Linux/macOS
# mrag-bench-env\Scripts\activate   # Windows

# 4. Install dependencies (slower)
pip install -r requirements.txt
```

### Verify Installation

**Always run in virtual environment:**

```bash
# Ensure virtual environment is activated
source mrag-bench-env/bin/activate

# Test basic setup
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test MRAG-Bench imports
python -c "from src.config import MRAGConfig; print('✅ Config loaded')"
python -c "from src.utils.memory_manager import MemoryMonitor; print('✅ Memory manager loaded')"
```

## 🧪 Testing

**Always run tests in virtual environment:**

```bash
# Activate environment first
source mrag-bench-env/bin/activate

# Run test suite
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_config.py -v
python -m pytest tests/test_memory_manager.py -v

# Check test coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## 🏗️ Development Commands

**All development commands require virtual environment:**

### Sprint Implementation

```bash
# Activate environment
source mrag-bench-env/bin/activate

# Run sprint implementations
/implement sprint-1    # Foundation setup
/implement sprint-2    # Dataset processing
/implement sprint-3    # CLIP retrieval
```

### Custom Commands

```bash
# Activate environment first
source mrag-bench-env/bin/activate

# Product planning
/plan-and-analyze "multimodal RAG system"

# Technical architecture
/architect

# Sprint planning
/sprint-plan
```

## 📁 Project Structure

```
mmrag-cs6101/
├── mrag-bench-env/           # 🔴 Virtual environment (REQUIRED)
├── src/
│   ├── dataset/              # MRAG-Bench dataset processing
│   ├── retrieval/            # CLIP-based image retrieval
│   ├── generation/           # LLaVA-based answer generation
│   ├── evaluation/           # MRAG-Bench evaluation framework
│   ├── utils/                # Memory management & utilities
│   └── config.py             # Configuration management
├── config/
│   └── mrag_bench.yaml       # System configuration
├── docs/
│   ├── prd.md                # Product Requirements Document
│   ├── sdd.md                # Software Design Document
│   ├── sprint.md             # Sprint Implementation Plan
│   └── a2a/                  # Implementation reports
├── tests/                    # Comprehensive test suite
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Hardware Optimization

For **RTX 5070Ti (16GB)**:

```bash
# Activate environment
source mrag-bench-env/bin/activate

# Test memory configuration
python -c "
from src.utils.memory_manager import MemoryMonitor
monitor = MemoryMonitor(memory_limit_gb=15.0)
stats = monitor.get_current_stats()
print(f'GPU: {stats.gpu_total_gb:.1f}GB available')
"
```

### Model Configuration

```bash
# Activate environment
source mrag-bench-env/bin/activate

# Check model configuration
python -c "
from src.config import MRAGConfig
config = MRAGConfig()
print(f'VLM: {config.model.vlm_name}')
print(f'Retriever: {config.model.retriever_name}')
print(f'Quantization: {config.model.quantization}')
print(f'Memory limit: {config.model.max_memory_gb}GB')
"
```

## 📊 Sprint Progress

| Sprint | Status | Focus | Deliverables |
|--------|--------|-------|--------------|
| Sprint 1 | ✅ Complete | Foundation | Interfaces, config, memory mgmt |
| Sprint 2 | ✅ Complete | Dataset | MRAG-Bench data processing |
| Sprint 3 | 🔄 Ready | Retrieval | CLIP image retrieval |
| Sprint 4 | ⏳ Pending | Generation | LLaVA integration |
| Sprint 5+ | ⏳ Pending | Optimization | Performance tuning |

## 🎯 Success Metrics

- **Target Accuracy**: 53-59% on perspective change scenarios
- **Memory Constraint**: ≤15GB VRAM usage
- **Processing Time**: <30 seconds per query
- **Reliability**: >99% successful inference completion

## 🛠️ Development Workflow

### Starting Development Session

```bash
# 1. Always activate virtual environment first
source mrag-bench-env/bin/activate

# 2. Verify environment is working
python -c "import src; print('✅ Environment ready')"

# 3. Run development commands
python your_script.py
pytest tests/
```

### Virtual Environment Management

```bash
# Create new environment (if needed)
uv venv mrag-bench-env --python 3.10

# Activate environment (required for all operations)
source mrag-bench-env/bin/activate

# Deactivate when done
deactivate

# Install additional packages (in activated environment)
uv pip install package-name

# Update requirements
uv pip freeze > requirements.txt
```

## ⚠️ Important Notes

1. **Always use virtual environment** - All Python operations must be run within `mrag-bench-env`
2. **GPU Memory Management** - System optimized for 16GB VRAM constraint
3. **Fresh Implementation** - This is a new system, not the legacy medical RAG
4. **Sprint-based Development** - Follow sprint sequence for proper implementation

## 🆘 Troubleshooting

### Virtual Environment Issues

```bash
# Environment not activated
# ❌ Wrong: python -c "import src"
# ✅ Correct:
source mrag-bench-env/bin/activate
python -c "import src"

# Missing dependencies
# ✅ Solution:
source mrag-bench-env/bin/activate
uv pip install -r requirements.txt
```

### Import Errors

```bash
# ModuleNotFoundError
# ✅ Solution: Always activate environment first
source mrag-bench-env/bin/activate
python your_script.py
```

### CUDA/GPU Issues

```bash
# Check GPU in virtual environment
source mrag-bench-env/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📖 Documentation

- **[PRD](docs/prd.md)** - Product requirements and success criteria
- **[SDD](docs/sdd.md)** - Software design and architecture
- **[Sprint Plan](docs/sprint.md)** - Implementation timeline and tasks
- **[Implementation Reports](docs/a2a/)** - Detailed sprint completion reports

## 🤝 Contributing

1. **Always use virtual environment**:
   ```bash
   source mrag-bench-env/bin/activate
   ```

2. **Run tests before committing**:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Follow sprint-based development**:
   ```bash
   /implement sprint-{n}
   ```

---

**⚠️ Remember: All development operations require the virtual environment to be activated first!**

```bash
# Always start with this
source mrag-bench-env/bin/activate
```