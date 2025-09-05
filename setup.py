#!/usr/bin/env python3
"""
Setup script for Medical Multimodal RAG System with uv
Fast development environment setup and dependency management
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional

def run_command(cmd: str, check: bool = True, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a shell command and return the result"""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        check=check, 
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0 and check:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result

def check_uv_installed() -> bool:
    """Check if uv is installed"""
    try:
        result = run_command("uv --version", check=False)
        if result.returncode == 0:
            print(f"✅ uv is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    return False

def install_uv():
    """Install uv package manager"""
    print("📦 Installing uv...")
    
    system = platform.system().lower()
    
    if system == "windows":
        # Windows installation
        run_command("pip install uv")
    else:
        # Unix-like systems (Linux, macOS)
        run_command("curl -LsSf https://astral.sh/uv/install.sh | sh")
        
        # Add to PATH for current session
        uv_bin = Path.home() / ".cargo" / "bin"
        if uv_bin.exists():
            os.environ["PATH"] = str(uv_bin) + os.pathsep + os.environ.get("PATH", "")
    
    print("✅ uv installed successfully")

def setup_project():
    """Setup the project with uv"""
    project_root = Path(__file__).parent
    
    print("🏗️  Setting up Medical Multimodal RAG project...")
    print(f"📁 Project root: {project_root}")
    
    # Navigate to project directory
    os.chdir(project_root)
    
    # Initialize uv project (if not already done)
    print("🔄 Initializing uv project...")
    run_command("uv init --no-readme", check=False)  # Don't fail if already initialized
    
    # Create virtual environment
    print("🐍 Creating virtual environment with uv...")
    run_command("uv venv")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    run_command("uv pip install -e .")
    
    # Install GPU support if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 CUDA detected, installing GPU support...")
            run_command("uv pip install faiss-gpu --force-reinstall")
            print("✅ GPU support installed")
    except ImportError:
        print("⚠️  PyTorch not yet installed, GPU support will be checked later")
    
    # Install development dependencies
    print("🛠️  Installing development dependencies...")
    run_command("uv pip install -e .[dev]")
    
    print("✅ Project setup completed!")

def setup_development_tools():
    """Setup development tools and pre-commit hooks"""
    print("🔧 Setting up development tools...")
    
    # Install pre-commit hooks
    try:
        run_command("pre-commit install")
        print("✅ Pre-commit hooks installed")
    except subprocess.CalledProcessError:
        print("⚠️  Pre-commit hooks installation failed (optional)")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = """# Medical RAG Environment Variables
TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0,1,2,3
TRANSFORMERS_CACHE=./cache/transformers
HF_HOME=./cache/huggingface
WANDB_DISABLED=true
"""
        env_file.write_text(env_content)
        print("✅ .env file created")

def run_tests():
    """Run the test suite to verify installation"""
    print("🧪 Running test suite...")
    
    try:
        # Set environment variables for testing
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        
        result = subprocess.run(
            [sys.executable, "test_medical_rag.py"],
            env=env,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"⚠️  Test execution failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("🎉 Medical Multimodal RAG Setup Complete!")
    print("="*60)
    
    print("\n📚 Quick Start:")
    print("1. Activate the environment:")
    
    system = platform.system().lower()
    if system == "windows":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n2. Test the system:")
    print("   python test_medical_rag.py")
    
    print("\n3. Install additional packages:")
    print("   uv add <package-name>")
    
    print("\n4. Run with GPU support:")
    print("   uv pip install -e .[gpu]")
    
    print("\n5. Start development:")
    print("   uv pip install -e .[dev]")
    
    print("\n🔧 Useful uv commands:")
    print("   uv add <package>          # Add a dependency")
    print("   uv remove <package>       # Remove a dependency") 
    print("   uv pip list               # List installed packages")
    print("   uv pip freeze             # Export requirements")
    print("   uv run <command>          # Run command in venv")
    print("   uv pip sync               # Sync dependencies")
    
    print("\n📖 Next Steps:")
    print("   • Download medical datasets (VQA-Med, PathVQA)")
    print("   • Build knowledge base with real medical data")
    print("   • Develop web demo interface")
    print("   • Set up distributed training on 4x RTX 2080Ti")
    
    print(f"\n📁 Project structure created at: {Path.cwd()}")

def main():
    """Main setup function"""
    print("🏥 Medical Multimodal RAG - Fast Setup with uv")
    print("=" * 50)
    
    # Check and install uv if needed
    if not check_uv_installed():
        install_uv()
    
    # Setup the project
    setup_project()
    
    # Setup development tools
    setup_development_tools()
    
    # Run tests to verify installation
    tests_passed = run_tests()
    
    # Print usage instructions
    print_usage_instructions()
    
    if tests_passed:
        print("\n🚀 System ready for development!")
        sys.exit(0)
    else:
        print("\n⚠️  Setup completed but some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()