#!/bin/bash
# Medical RAG Development Workflow with uv
# Fast commands for common development tasks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'  
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏥 Medical Multimodal RAG - Development Workflow${NC}"
echo "=================================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv not found. Please install uv first.${NC}"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${GREEN}✅ uv found: $(uv --version)${NC}"

# Function to run commands with status
run_with_status() {
    echo -e "${BLUE}🔧 $1${NC}"
    if eval "$2"; then
        echo -e "${GREEN}✅ $1 completed${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-help}" in
    "setup"|"install")
        echo -e "${YELLOW}📦 Setting up development environment...${NC}"
        run_with_status "Creating virtual environment" "uv venv"
        run_with_status "Installing project dependencies" "uv pip install -e ."
        run_with_status "Installing development dependencies" "uv pip install -e .[dev]"
        
        # Check for CUDA and install GPU support
        if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            echo -e "${YELLOW}🚀 CUDA detected, installing GPU support...${NC}"
            run_with_status "Installing GPU support" "uv add faiss-gpu --force-reinstall"
        else
            echo -e "${YELLOW}ℹ️  No CUDA detected, skipping GPU support${NC}"
        fi
        
        echo -e "${GREEN}🎉 Development environment ready!${NC}"
        ;;
        
    "test")
        echo -e "${YELLOW}🧪 Running test suite...${NC}"
        run_with_status "Running medical RAG tests" "uv run python test_medical_rag.py"
        ;;
        
    "format")
        echo -e "${YELLOW}🎨 Formatting code...${NC}"
        run_with_status "Running black formatter" "uv run black src/ tests/ *.py"
        run_with_status "Running isort import sorter" "uv run isort src/ tests/ *.py"
        ;;
        
    "lint")
        echo -e "${YELLOW}🔍 Linting code...${NC}"
        run_with_status "Running flake8 linter" "uv run flake8 src/ tests/ --max-line-length=100"
        ;;
        
    "type-check")
        echo -e "${YELLOW}🔎 Type checking...${NC}"
        run_with_status "Running mypy type checker" "uv run mypy src/ --ignore-missing-imports"
        ;;
        
    "gpu")
        echo -e "${YELLOW}🚀 Installing/updating GPU support...${NC}"
        run_with_status "Installing faiss-gpu" "uv add faiss-gpu --force-reinstall"
        run_with_status "Verifying GPU setup" "uv run python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\"); import faiss; print(f\"FAISS GPUs: {faiss.get_num_gpus()}\")'"
        ;;
        
    "clean")
        echo -e "${YELLOW}🧹 Cleaning project...${NC}"
        rm -rf .venv/
        rm -rf build/
        rm -rf dist/
        rm -rf *.egg-info/
        rm -rf src/*.egg-info/
        find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete 2>/dev/null || true
        echo -e "${GREEN}✅ Project cleaned${NC}"
        ;;
        
    "demo")
        echo -e "${YELLOW}🌐 Starting demo (when implemented)...${NC}"
        # This will be implemented later
        echo -e "${BLUE}ℹ️  Demo interface not yet implemented${NC}"
        echo -e "${BLUE}ℹ️  Use 'uv run python test_medical_rag.py' to test the system${NC}"
        ;;
        
    "requirements")
        echo -e "${YELLOW}📋 Exporting requirements...${NC}"
        run_with_status "Exporting requirements.txt" "uv pip freeze > requirements-uv.txt"
        echo -e "${GREEN}✅ Requirements exported to requirements-uv.txt${NC}"
        ;;
        
    "update")
        echo -e "${YELLOW}⬆️  Updating dependencies...${NC}"
        run_with_status "Updating all dependencies" "uv pip sync"
        ;;
        
    "benchmark")
        echo -e "${YELLOW}⏱️  Running performance benchmarks...${NC}"
        run_with_status "Benchmarking system" "uv run python -c '
import time
from src.medical_rag import MedicalMultimodalRAG
from PIL import Image

print(\"🏥 Medical RAG Performance Benchmark\")
print(\"=\"*40)

# Initialize system
start = time.time()
rag = MedicalMultimodalRAG()
init_time = time.time() - start
print(f\"Initialization: {init_time:.2f}s\")

# Test query
dummy_img = Image.new(\"RGB\", (224, 224), color=\"white\")
start = time.time()
# This will fail without knowledge base, but measures initialization speed
print(f\"System ready in: {init_time:.2f} seconds\")
'"
        ;;
        
    "help"|*)
        echo -e "${BLUE}Available commands:${NC}"
        echo ""
        echo -e "${GREEN}setup${NC}      - Set up development environment with uv"
        echo -e "${GREEN}test${NC}       - Run comprehensive test suite" 
        echo -e "${GREEN}format${NC}     - Format code with black and isort"
        echo -e "${GREEN}lint${NC}       - Lint code with flake8"
        echo -e "${GREEN}type-check${NC} - Run mypy type checking"
        echo -e "${GREEN}gpu${NC}        - Install/update GPU support (faiss-gpu)"
        echo -e "${GREEN}clean${NC}      - Clean build artifacts and cache"
        echo -e "${GREEN}demo${NC}       - Start web demo interface"
        echo -e "${GREEN}requirements${NC} - Export current dependencies"
        echo -e "${GREEN}update${NC}     - Update all dependencies"
        echo -e "${GREEN}benchmark${NC}  - Run performance benchmarks"
        echo ""
        echo -e "${BLUE}Example usage:${NC}"
        echo -e "${YELLOW}  ./scripts/dev-workflow.sh setup${NC}     # First-time setup"
        echo -e "${YELLOW}  ./scripts/dev-workflow.sh test${NC}      # Run tests"
        echo -e "${YELLOW}  ./scripts/dev-workflow.sh format${NC}    # Format code"
        echo -e "${YELLOW}  ./scripts/dev-workflow.sh gpu${NC}       # Install GPU support"
        echo ""
        echo -e "${BLUE}Speed comparison (uv vs pip):${NC}"
        echo -e "${GREEN}  uv: ~10-100x faster for dependency resolution${NC}"
        echo -e "${GREEN}  uv: ~5-10x faster for installations${NC}"
        ;;
esac

echo -e "${BLUE}================================================${NC}"