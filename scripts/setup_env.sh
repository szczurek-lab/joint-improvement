#!/usr/bin/env bash
set -euo pipefail
ENV_NAME=${1:-joint-improvement}

# Verify micromamba is installed and available
if ! command -v micromamba &> /dev/null; then
  echo "Error: micromamba is not installed or not in PATH."
  echo "Please install micromamba from: https://mamba.readthedocs.io/en/latest/installation.html"
  exit 1
fi

# Check for C compiler (required for PyTorch compilation with torch.compile)
if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
  echo "C compiler not found. Attempting to install build-essential..."
  if command -v apt-get &> /dev/null; then
    # Try with fix-missing first, then try without if that fails
    if sudo apt-get update 2>/dev/null && sudo apt-get install -y --fix-missing build-essential 2>/dev/null; then
      echo "C compiler installed successfully."
    elif sudo apt-get install -y gcc g++ make 2>/dev/null; then
      echo "C compiler (gcc/g++) installed successfully."
    else
      echo "Warning: Failed to install C compiler automatically (repository issues)."
      echo "Please try manually:"
      echo "  sudo apt-get update"
      echo "  sudo apt-get install -y build-essential"
      echo "Or install individual packages: sudo apt-get install -y gcc g++ make"
      echo ""
      echo "If you encounter 404 errors, you may need to update package mirrors."
    fi
  elif command -v yum &> /dev/null; then
    if sudo yum install -y gcc gcc-c++ make; then
      echo "C compiler installed successfully."
    else
      echo "Warning: Failed to install C compiler automatically."
      echo "Please install manually: sudo yum install -y gcc gcc-c++ make"
    fi
  elif command -v brew &> /dev/null; then
    if brew install gcc; then
      echo "C compiler installed successfully."
    else
      echo "Warning: Failed to install C compiler automatically."
      echo "Please install manually: brew install gcc"
    fi
  else
    echo "Warning: Could not automatically install C compiler."
    echo "Please install gcc or clang manually for torch.compile support."
    echo "Ubuntu/Debian: sudo apt-get install build-essential"
    echo "CentOS/RHEL: sudo yum install gcc gcc-c++ make"
    echo "macOS: brew install gcc"
  fi
else
  echo "C compiler found: $(command -v gcc || command -v clang)"
fi

if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  micromamba create -f environment.yml -n "$ENV_NAME"
fi

# Install PyTorch with CUDA 12.6 FIRST (before packages that depend on it)
micromamba run -n "$ENV_NAME" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install project and all dependencies (pip will see PyTorch is already installed)
# This installs: transformers, datasets, accelerate, and dev dependencies
micromamba run -n "$ENV_NAME" pip install -e .[dev]
micromamba run -n "$ENV_NAME" pre-commit install

# Verify installation
if micromamba run -n "$ENV_NAME" python -c "
import sys
import importlib

# Check package installation
try:
    import joint_improvement
    print('joint-improvement package installed')
except ImportError as e:
    print(f'Failed to import joint-improvement: {e}')
    sys.exit(1)

# Check PyTorch
try:
    import torch
    print(f'PyTorch {torch.__version__} installed')
    if torch.cuda.is_available():
        print(f'CUDA available: {torch.version.cuda}')
    else:
        print('CUDA not available (CPU-only mode)')
except ImportError as e:
    print(f'Failed to import PyTorch: {e}')
    sys.exit(1)
"; then
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Installation verification failed!"
    echo "=========================================="
    exit 1
fi
