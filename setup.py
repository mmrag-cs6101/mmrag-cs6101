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
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        # Show output if there's useful information
        if result.stdout and result.stdout.strip():
            print(f"   ✓ {result.stdout.strip()}")
        
        return result
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"   Return code: {e.returncode}")
        if e.stdout:
            print(f"   Stdout: {e.stdout.strip()}")
        if e.stderr:
            print(f"   Stderr: {e.stderr.strip()}")
        
        # Provide specific guidance for common errors
        if "uv" in cmd and ("not found" in str(e) or e.returncode == 127):
            print("\n💡 uv installation issue detected:")
            print("   Try: pip install uv")
            print("   Or: curl -LsSf https://astral.sh/uv/install.sh | sh")
        elif "venv" in cmd:
            print("\n💡 Virtual environment creation failed:")
            print("   Check if Python is properly installed")
            print(f"   Python version: {sys.version}")
        
        if check:
            sys.exit(1)
        return e
    
    except Exception as e:
        print(f"❌ Unexpected error running command: {cmd}")
        print(f"   Error: {str(e)}")
        if check:
            sys.exit(1)
        return None

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
    
    # Create virtual environment (clear existing if present)
    print("🐍 Creating virtual environment with uv...")
    venv_path = Path(".venv")
    if venv_path.exists():
        print("   ℹ️  Existing .venv found, recreating...")
        run_command("uv venv")
    else:
        run_command("uv venv")
    
    # Install package first (without dependencies to avoid timeout)
    print("📦 Installing package structure...")
    run_command("uv pip install --no-deps -e .")
    
    # Install core dependencies in batches to avoid timeout
    print("📦 Installing core ML dependencies (this may take a few minutes)...")
    
    # Install PyTorch first (largest dependency)
    print("   🔧 Installing PyTorch...")
    run_command("uv add torch torchvision torchaudio", check=False)
    
    # Install transformers
    print("   🔧 Installing Transformers...")
    run_command("uv add transformers accelerate", check=False)
    
    # Install other dependencies
    print("   🔧 Installing other dependencies...")
    run_command("uv add pillow numpy pandas scikit-learn", check=False)
    run_command("uv add faiss-cpu sentence-transformers", check=False)
    run_command("uv add gradio streamlit fastapi uvicorn", check=False)
    
    # Check for CUDA and install GPU support
    print("🔍 Checking for CUDA support...")
    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 CUDA detected, installing GPU support...")
            run_command("uv add faiss-gpu --force-reinstall", check=False)
            print("✅ GPU support installed")
        else:
            print("ℹ️  CUDA not detected, using CPU-only versions")
    except ImportError:
        print("⚠️  PyTorch not yet available, skipping GPU check")
        print("   💡 Run './scripts/dev-workflow.sh gpu' later to add GPU support")
    
    # Install development dependencies
    print("🛠️  Installing development dependencies...")
    run_command("uv add --dev pytest black isort flake8 mypy pre-commit", check=False)
    
    print("✅ Project setup completed!")

def setup_development_tools():
    """Setup development tools and pre-commit hooks"""
    print("🔧 Setting up development tools...")
    
    # Install pre-commit hooks (only if pre-commit is available)
    try:
        # First check if pre-commit is installed
        result = run_command("uv run pre-commit --version", check=False)
        if result and result.returncode == 0:
            run_command("uv run pre-commit install")
            print("✅ Pre-commit hooks installed")
        else:
            print("⚠️  Pre-commit not available, skipping hooks setup")
            print("   💡 Run 'uv add --dev pre-commit' then 'uv run pre-commit install'")
    except Exception:
        print("⚠️  Pre-commit hooks installation skipped (pre-commit not installed yet)")
    
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