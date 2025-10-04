#!/bin/bash
# MRAG-Bench System Verification Script
# Runs quick checks to verify the complete system is working

set -e  # Exit on error

echo "=================================="
echo "MRAG-Bench System Verification"
echo "=================================="
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "❌ Error: Virtual environment not activated"
    echo "Please run: source mrag-bench-env/bin/activate"
    exit 1
fi

echo "✓ Virtual environment activated: $VIRTUAL_ENV"
echo ""

# Step 1: Health Check
echo "Step 1/4: Running Health Check..."
echo "=================================="
python src/utils/health_check.py
if [ $? -ne 0 ]; then
    echo "❌ Health check failed. Please fix issues before proceeding."
    exit 1
fi
echo ""

# Step 2: Verify imports
echo "Step 2/4: Verifying System Imports..."
echo "=================================="
python -c "
import sys
sys.path.insert(0, 'src')

print('Testing core imports...')
from src.config import MRAGConfig
from src.dataset import MRAGDataset
from src.retrieval import CLIPRetriever
from src.generation import LLaVAGenerationPipeline
from src.pipeline import MRAGPipeline
from src.evaluation import MRAGBenchEvaluator
print('✓ All core imports successful')

print('\\nChecking configuration...')
config = MRAGConfig()
print(f'✓ Config loaded: {config.model.vlm_name}')
print(f'✓ Device: {config.model.device}')
"
if [ $? -ne 0 ]; then
    echo "❌ Import verification failed"
    exit 1
fi
echo ""

# Step 3: Check dataset
echo "Step 3/4: Checking Dataset..."
echo "=================================="
python -c "
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import json

dataset_path = Path('data/mrag_bench')
if not dataset_path.exists():
    print('❌ Dataset not found at data/mrag_bench')
    print('Please run: python download_mrag_dataset.py')
    sys.exit(1)

# Check metadata
metadata_file = dataset_path / 'metadata' / 'dataset_info.json'
if metadata_file.exists():
    with open(metadata_file) as f:
        info = json.load(f)
    print(f'✓ Dataset found: {info[\"total_samples\"]} samples')
    print(f'✓ Scenarios: {list(info[\"scenarios\"].keys())}')
else:
    print('⚠ Dataset metadata not found, but directory exists')
"
if [ $? -ne 0 ]; then
    echo "❌ Dataset check failed"
    exit 1
fi
echo ""

# Step 4: Quick pipeline test
echo "Step 4/4: Quick Pipeline Test..."
echo "=================================="
echo "Running quick evaluation test (this may take 5-10 minutes)..."
python run_sprint10_final_validation.py --quick-test

if [ $? -ne 0 ]; then
    echo "❌ Pipeline test failed"
    exit 1
fi
echo ""

# Success!
echo "=================================="
echo "✅ System Verification Complete!"
echo "=================================="
echo ""
echo "Your MRAG-Bench system is ready to use!"
echo ""
echo "Next steps:"
echo "  • Run full evaluation: python run_sprint10_final_validation.py --num-runs 3"
echo "  • See documentation: docs/IMPLEMENTATION_GUIDE.md"
echo "  • Check API docs: docs/API_DOCUMENTATION.md"
echo ""
