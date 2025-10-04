#!/bin/bash
# MRAG-Bench Environment Setup Script
# Version: 1.0
# Date: October 4, 2025

set -e  # Exit on error

echo "============================================"
echo "MRAG-Bench Environment Setup"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="mrag-bench-env"
PYTHON_VERSION="3.10"
CUDA_VERSION="118"  # 11.8

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running in WSL or Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_status "Detected Linux environment"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    print_status "Detected Windows environment"
else
    print_warning "Detected macOS environment (limited GPU support)"
fi

# Step 1: Check Python version
echo ""
echo "Step 1: Checking Python version..."
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    print_status "Python 3.10 found"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if [[ "$PYTHON_VER" == "3.8" ]] || [[ "$PYTHON_VER" == "3.9" ]] || [[ "$PYTHON_VER" == "3.10" ]] || [[ "$PYTHON_VER" == "3.11" ]]; then
        print_status "Python $PYTHON_VER found"
    else
        print_error "Python 3.8+ required, found $PYTHON_VER"
        exit 1
    fi
else
    print_error "Python 3.8+ not found. Please install Python 3.10"
    exit 1
fi

# Step 2: Check for UV package manager
echo ""
echo "Step 2: Checking for UV package manager..."
if command -v uv &> /dev/null; then
    print_status "UV package manager found"
    USE_UV=true
else
    print_warning "UV not found. Will use standard pip (slower)"
    print_warning "Install UV for 10x faster package installation:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    USE_UV=false
fi

# Step 3: Create virtual environment
echo ""
echo "Step 3: Creating virtual environment..."
if [ -d "$ENV_NAME" ]; then
    print_warning "Virtual environment '$ENV_NAME' already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$ENV_NAME"
        print_status "Removed existing environment"
    else
        print_status "Using existing environment"
    fi
fi

if [ ! -d "$ENV_NAME" ]; then
    if [ "$USE_UV" = true ]; then
        uv venv "$ENV_NAME" --python "$PYTHON_VERSION"
    else
        $PYTHON_CMD -m venv "$ENV_NAME"
    fi
    print_status "Virtual environment created: $ENV_NAME"
fi

# Step 4: Activate virtual environment
echo ""
echo "Step 4: Activating virtual environment..."
source "$ENV_NAME/bin/activate"
print_status "Virtual environment activated"

# Step 5: Upgrade pip (if not using UV)
if [ "$USE_UV" = false ]; then
    echo ""
    echo "Step 5: Upgrading pip..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    print_status "Pip upgraded"
fi

# Step 6: Install PyTorch with CUDA
echo ""
echo "Step 6: Installing PyTorch with CUDA $CUDA_VERSION..."
print_warning "This may take several minutes..."

if [ "$USE_UV" = true ]; then
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$CUDA_VERSION
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$CUDA_VERSION
fi
print_status "PyTorch with CUDA installed"

# Step 7: Install project dependencies
echo ""
echo "Step 7: Installing project dependencies..."
print_warning "This may take several minutes..."

if [ "$USE_UV" = true ]; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi
print_status "Dependencies installed"

# Step 8: Verify installation
echo ""
echo "Step 8: Verifying installation..."

# Check PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    python -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else 'N/A')"
    print_status "CUDA verification successful"
else
    print_warning "CUDA not available - GPU acceleration disabled"
fi

# Check MRAG-Bench system
if python -c "from src.config import MRAGConfig; from src.utils.memory_manager import MemoryMonitor; print('MRAG-Bench system ready')" 2>&1 | grep -q "ready"; then
    print_status "MRAG-Bench system imports successful"
else
    print_error "MRAG-Bench system imports failed"
    exit 1
fi

# Step 9: Create necessary directories
echo ""
echo "Step 9: Creating necessary directories..."
mkdir -p data/mrag_bench
mkdir -p data/embeddings
mkdir -p output
mkdir -p results
mkdir -p logs
print_status "Directories created"

# Step 10: Download models (optional)
echo ""
echo "Step 10: Pre-downloading models (optional)..."
read -p "Download LLaVA-1.5-7B and CLIP models now? (~15GB, y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Downloading models... This will take 10-30 minutes..."
    python -c "
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import CLIPModel, CLIPProcessor

print('Downloading LLaVA-1.5-7B...')
model = LlavaNextForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf')
processor = LlavaNextProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
print('✓ LLaVA-1.5-7B downloaded')

print('Downloading CLIP ViT-B/32...')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('✓ CLIP ViT-B/32 downloaded')
print('✓ All models downloaded successfully')
"
    print_status "Models pre-downloaded"
else
    print_status "Skipping model download (will download on first use)"
fi

# Final summary
echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
print_status "Virtual environment: $ENV_NAME"
print_status "Python version: $($PYTHON_CMD --version)"
print_status "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
print_status "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source $ENV_NAME/bin/activate"
echo "  2. Download dataset: python download_mrag_dataset.py"
echo "  3. Run tests: python -m pytest tests/ -v"
echo "  4. Run evaluation: python run_sprint7_mvp_evaluation.py --max-samples 50"
echo ""
echo "For help, see:"
echo "  - docs/IMPLEMENTATION_GUIDE.md"
echo "  - docs/TROUBLESHOOTING.md"
echo ""
