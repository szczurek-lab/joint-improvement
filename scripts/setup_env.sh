#!/usr/bin/env bash
set -euo pipefail
ENV_NAME=${1:-joint-improvement}

# Verify micromamba is installed and available
if ! command -v micromamba &> /dev/null; then
  echo "Error: micromamba is not installed or not in PATH."
  echo "Please install micromamba from: https://mamba.readthedocs.io/en/latest/installation.html"
  exit 1
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
