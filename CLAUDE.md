# MRAG-Bench Reproduction System - Claude Code Guide

This document provides comprehensive guidance for using Claude Code with the MRAG-Bench reproduction system.

## 🎯 Project Overview

This is a **fresh implementation** of a multimodal retrieval-augmented generation system designed to reproduce MRAG-Bench paper results. The system targets **53-59% accuracy** on perspective change scenarios using:

- **LLaVA-1.5-7B** (Vision-Language Model)
- **CLIP ViT-B/32** (Image Retriever)
- **4-bit quantization** for memory efficiency
- **16GB VRAM constraint** optimization

## 🚀 Quick Start with Claude Code

### Environment Setup (Required)

**⚠️ CRITICAL: Always use virtual environment for all operations**

```bash
# 1. Create virtual environment (UV recommended - 10x faster)
uv venv mrag-bench-env --python 3.10

# 2. Activate virtual environment (REQUIRED for all commands)
source mrag-bench-env/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Verify setup
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from src.config import MRAGConfig; print('✅ MRAG-Bench system ready')"
```

### Custom Claude Code Commands

This project includes custom commands for streamlined development:

```bash
# Product planning and requirements
/plan-and-analyze "multimodal RAG system features"

# Technical architecture design
/architect

# Sprint planning and task breakdown
/sprint-plan

# Sprint implementation with review cycle
/implement sprint-1    # Foundation setup
/implement sprint-2    # Dataset processing
/implement sprint-3    # CLIP retrieval
```

## 📋 Sprint-Based Development

### Current Sprint Status

| Sprint | Status | Focus | Duration | Deliverables |
|--------|--------|-------|----------|--------------|
| **Sprint 1** | ✅ Complete | Foundation | Days 1-2.5 | Interfaces, config, memory mgmt |
| **Sprint 2** | ✅ Complete | Dataset | Days 3-5.5 | MRAG-Bench data processing |
| **Sprint 3** | 🔄 Ready | Retrieval | Days 6-8.5 | CLIP image retrieval |
| **Sprint 4** | ⏳ Pending | Integration | Days 9-11.5 | End-to-end pipeline |
| **Sprint 5** | ⏳ Pending | Optimization | Days 12-14.5 | Performance tuning |

### Sprint Implementation

Each sprint follows a structured implementation and review cycle:

```bash
# Always activate environment first
source mrag-bench-env/bin/activate

# Run sprint implementation
/implement sprint-{number}

# Review implementation report
cat docs/a2a/reviewer.md

# Test implementation
python -m pytest tests/ -v
```

## 🏗️ System Architecture

### Core Components

```
MRAG-Bench System Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dataset       │    │    Retrieval     │    │   Generation    │
│   Interface     │──→ │    Pipeline      │──→ │    Pipeline     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ MRAG-Bench      │    │ CLIP ViT-B/32    │    │ LLaVA-1.5-7B    │
│ Data Loader     │    │ + FAISS Index    │    │ + 4-bit Quant   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Module Structure

```python
# Dataset Processing (Sprint 2 - Complete)
from src.dataset import MRAGDataset, DatasetInterface, ImagePreprocessor
from src.dataset import MemoryAwareDataLoader, DatasetValidator

# Retrieval System (Sprint 3 - In Progress)
from src.retrieval import RetrievalPipeline, CLIPRetriever
from src.retrieval import VectorStore, SimilaritySearch

# Generation System (Sprint 4 - Pending)
from src.generation import GenerationPipeline, LLaVAModel
from src.generation import QuantizedInference, ResponseGenerator

# Evaluation Framework (Sprint 5 - Pending)
from src.evaluation import MRAGBenchEvaluator, EvaluationResults
from src.evaluation import AccuracyMetrics, PerformanceTracker

# Utilities (Sprint 1 - Complete)
from src.utils import MemoryManager, ErrorHandler
from src.config import MRAGConfig
```

## 🧪 Testing and Validation

### Testing Requirements

**All tests must run in virtual environment:**

```bash
# Activate environment (REQUIRED)
source mrag-bench-env/bin/activate

# Run comprehensive test suite
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific component tests
python -m pytest tests/test_dataset.py -v      # Dataset processing
python -m pytest tests/test_memory_manager.py -v  # Memory management
python -m pytest tests/test_config.py -v      # Configuration system
```

### Validation Checklist

- ✅ **Environment Setup**: Virtual environment activated
- ✅ **GPU Detection**: CUDA available, RTX 5070Ti detected
- ✅ **Memory Management**: <15GB VRAM usage
- ✅ **Module Imports**: All core components import successfully
- ✅ **Configuration**: YAML config loads properly
- ✅ **Dataset Processing**: MRAG-Bench data pipeline operational

### Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| **Accuracy** | 53-59% | Sprint 3+ |
| **Memory Usage** | ≤15GB VRAM | ✅ <1GB (Sprint 2) |
| **Dataset Loading** | <30s | ✅ <5s (Sprint 2) |
| **Processing Time** | <30s per query | Sprint 4+ |
| **Test Coverage** | >90% | ✅ 96% (Sprint 2) |

## 🔧 Development Workflow

### Starting a Development Session

```bash
# 1. ALWAYS start by activating virtual environment
source mrag-bench-env/bin/activate

# 2. Verify environment is working
python -c "
import torch
from src.config import MRAGConfig
from src.utils.memory_manager import MemoryMonitor

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'✅ MRAG-Bench system ready')
"

# 3. Run development commands
```

### Code Development

```bash
# Always work in virtual environment
source mrag-bench-env/bin/activate

# Edit code with proper imports
python -c "
# Example: Test dataset functionality
from src.dataset import MRAGDataset
from src.config import MRAGConfig

config = MRAGConfig()
print(f'Dataset config: {config.dataset.data_path}')
"

# Run tests after changes
python -m pytest tests/test_your_changes.py -v
```

### Sprint Implementation

```bash
# Environment setup
source mrag-bench-env/bin/activate

# Check current sprint status
cat docs/sprint.md

# Implement next sprint
/implement sprint-3

# Review results
cat docs/a2a/reviewer.md
```

## 📁 File Structure

```
mmrag-cs6101/
├── mrag-bench-env/                 # 🔴 Virtual environment (REQUIRED)
│   ├── bin/activate               # Environment activation script
│   ├── lib/python3.10/           # Python packages
│   └── ...
├── .claude/                       # Claude Code configuration
│   ├── commands/                  # Custom commands
│   │   ├── plan-and-analyze.md    # Product planning
│   │   ├── architect.md           # Technical architecture
│   │   ├── sprint-plan.md         # Sprint planning
│   │   └── implement.md           # Sprint implementation
│   └── agents/                    # Specialized agents
├── src/                           # Core system implementation
│   ├── dataset/                   # MRAG-Bench dataset processing
│   │   ├── interface.py           # Abstract interfaces
│   │   ├── mrag_dataset.py        # Core dataset implementation
│   │   ├── preprocessing.py       # Image preprocessing
│   │   ├── data_loader.py         # Memory-aware loading
│   │   └── validation.py          # Dataset validation
│   ├── retrieval/                 # CLIP-based retrieval (Sprint 3)
│   ├── generation/                # LLaVA-based generation (Sprint 4)
│   ├── evaluation/                # MRAG-Bench evaluation (Sprint 5)
│   ├── utils/                     # Utilities and helpers
│   │   ├── memory_manager.py      # VRAM monitoring
│   │   ├── error_handling.py      # Error recovery
│   │   └── optimization.py        # Performance optimization
│   └── config.py                  # Configuration management
├── config/
│   └── mrag_bench.yaml            # System configuration
├── docs/                          # Documentation
│   ├── prd.md                     # Product Requirements Document
│   ├── sdd.md                     # Software Design Document
│   ├── sprint.md                  # Sprint Implementation Plan
│   └── a2a/                       # Agent-to-agent communication
│       ├── reviewer.md            # Implementation reports
│       └── engineer-feedback.md   # Review feedback
├── tests/                         # Comprehensive test suite
├── data/                          # Dataset storage (created at runtime)
├── requirements.txt               # Python dependencies
├── README.md                      # User documentation
└── CLAUDE.md                      # This file
```

## 🎯 Success Metrics and Goals

### Primary Objectives

1. **Accuracy Target**: Achieve 53-59% accuracy on MRAG-Bench perspective change scenarios
2. **Memory Efficiency**: Operate within 16GB VRAM constraint (target ≤15GB)
3. **Performance**: Complete queries in <30 seconds
4. **Reliability**: >99% successful inference completion rate

### Development Quality

- **Test Coverage**: Maintain >90% test coverage
- **Code Quality**: Follow Python best practices and type hints
- **Documentation**: Comprehensive inline and external documentation
- **Memory Management**: Proactive VRAM monitoring and optimization

### Sprint Success Criteria

Each sprint has specific deliverables and acceptance criteria:

- **Sprint 1**: ✅ Foundation interfaces and infrastructure
- **Sprint 2**: ✅ Dataset processing pipeline (exceeded targets)
- **Sprint 3**: CLIP retrieval system with vector indexing
- **Sprint 4**: LLaVA integration and end-to-end pipeline
- **Sprint 5**: Performance optimization and accuracy validation

## 🛠️ Troubleshooting

### Common Issues

#### Virtual Environment Not Activated

```bash
# ❌ Error: ModuleNotFoundError: No module named 'src'
# ✅ Solution:
source mrag-bench-env/bin/activate
python your_script.py
```

#### CUDA/GPU Issues

```bash
# Check GPU in virtual environment
source mrag-bench-env/bin/activate
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"
```

#### Memory Issues

```bash
# Monitor memory usage
source mrag-bench-env/bin/activate
python -c "
from src.utils.memory_manager import MemoryMonitor
monitor = MemoryMonitor(memory_limit_gb=15.0)
stats = monitor.get_current_stats()
print(f'GPU Memory: {stats.gpu_allocated_gb:.1f}/{stats.gpu_total_gb:.1f}GB')
print(f'Utilization: {stats.gpu_utilization_percent():.1f}%')
"
```

#### Import Errors

```bash
# Fix import issues
source mrag-bench-env/bin/activate
python -c "
# Test all core imports
try:
    from src.config import MRAGConfig
    from src.dataset import DatasetInterface
    from src.utils.memory_manager import MemoryMonitor
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

### Debug Commands

```bash
# Full system verification
source mrag-bench-env/bin/activate
python -c "
print('🔍 MRAG-Bench System Debug')
print('=' * 40)

# Environment
import sys
print(f'Python: {sys.version}')

# PyTorch
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

# System components
from src.config import MRAGConfig
from src.utils.memory_manager import MemoryMonitor

config = MRAGConfig()
monitor = MemoryMonitor()
stats = monitor.get_current_stats()

print(f'Model: {config.model.vlm_name}')
print(f'Memory: {stats.gpu_allocated_gb:.1f}/{stats.gpu_total_gb:.1f}GB')
print('✅ System ready for development')
"
```

## 📖 Best Practices

### Development Guidelines

1. **Always use virtual environment**: Every Python command must run in `mrag-bench-env`
2. **Follow sprint sequence**: Implement sprints in order (1→2→3→4→5)
3. **Test after changes**: Run relevant tests after any code modifications
4. **Monitor memory**: Use memory manager for VRAM-intensive operations
5. **Document changes**: Update relevant documentation for significant changes

### Code Standards

```python
# Example code structure
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class YourClass:
    """Clear docstring describing the class."""
    param: str
    optional_param: Optional[int] = None

    def your_method(self, input_data: List[str]) -> Dict[str, Any]:
        """
        Clear method documentation.

        Args:
            input_data: Description of input parameter

        Returns:
            Description of return value
        """
        try:
            # Implementation with proper error handling
            result = self._process_data(input_data)
            logger.info(f"Processed {len(input_data)} items")
            return result
        except Exception as e:
            logger.error(f"Error in your_method: {e}")
            raise
```

### Memory Management

```python
# Use memory manager for GPU operations
from src.utils.memory_manager import MemoryManager

manager = MemoryManager(memory_limit_gb=15.0)

# Check memory before operations
if manager.check_memory_availability(required_gb=5.0):
    # Proceed with GPU operation
    pass
else:
    # Handle memory constraint
    manager.clear_gpu_memory()
```

## 🚀 Next Steps

### Immediate Actions

1. **Activate virtual environment**: `source mrag-bench-env/bin/activate`
2. **Verify setup**: Run verification commands above
3. **Review current sprint**: Check `docs/sprint.md` for status
4. **Continue development**: Run `/implement sprint-3` for next phase

### Long-term Goals

- **Sprint 3**: Implement CLIP-based image retrieval with FAISS indexing
- **Sprint 4**: Integrate LLaVA-1.5-7B with quantization
- **Sprint 5**: Optimize performance and validate accuracy targets
- **Beyond**: Extend to other MRAG-Bench scenarios and evaluations

---

**⚠️ Remember: Virtual environment activation is required for ALL operations!**

```bash
# Always start with this
source mrag-bench-env/bin/activate
```

This ensures consistent behavior and proper dependency management throughout the development process.