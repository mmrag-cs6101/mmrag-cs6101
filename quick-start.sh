#!/bin/bash
# Quick Start Script for Medical Multimodal RAG
# One command to get everything running

set -e

echo "🏥 Medical Multimodal RAG - Quick Start"
echo "======================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        pip install uv
    else
        # Linux/macOS
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env 2>/dev/null || true
    fi
    echo "✅ uv installed!"
fi

# Run the main setup
echo "🚀 Running automated setup..."
python setup.py

echo ""
echo "🎉 Setup complete! Your Medical RAG system is ready."
echo ""
echo "🔗 Useful commands:"
echo "  ./scripts/dev-workflow.sh test    # Run tests" 
echo "  ./scripts/dev-workflow.sh gpu     # Install GPU support"
echo "  ./scripts/dev-workflow.sh help    # See all commands"
echo ""
echo "📚 Next steps:"
echo "  1. Download medical datasets"
echo "  2. Build knowledge base"
echo "  3. Start developing!"