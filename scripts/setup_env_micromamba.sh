#!/usr/bin/env bash
set -euo pipefail

# Parse command-line arguments
ENV_DIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--prefix)
      if [ -z "${2:-}" ]; then
        echo "Error: --prefix/-p requires a path argument"
        echo "Usage: $0 [--prefix|-p PATH]"
        exit 1
      fi
      ENV_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--prefix|-p PATH]"
      echo "  --prefix, -p PATH   Optional: Path where the environment should be created"
      exit 1
      ;;
  esac
done

# Verify environment.yml exists
ENV_YML="environment.yml"
if [ ! -f "$ENV_YML" ]; then
  echo "Error: environment.yml not found at: $ENV_YML"
  exit 1
fi

# Extract environment name from environment.yml (used when NOT using --prefix)
ENV_NAME="$(grep -E "^name:" "$ENV_YML" | sed 's/^name:[[:space:]]*//' | sed 's/[[:space:]]*$//')"
if [ -z "$ENV_NAME" ]; then
  echo "Error: Could not extract environment name from environment.yml"
  echo "Please ensure environment.yml contains a 'name:' field."
  exit 1
fi
if [[ "$ENV_NAME" == *"/"* ]]; then
  echo "Error: Invalid environment name in environment.yml: '$ENV_NAME'"
  exit 1
fi

# Verify micromamba is installed and available
if ! command -v micromamba &> /dev/null; then
  echo "Error: micromamba is not installed or not in PATH."
  echo "Please install micromamba from: https://mamba.readthedocs.io/en/latest/installation.html"
  exit 1
fi

# Set environment specification: prefix if provided, otherwise use name from environment.yml
if [ -n "$ENV_DIR" ]; then
  ENV_SPEC="-p"
  ENV_LOCATION="$ENV_DIR"
  if [ -d "$ENV_DIR" ]; then
    # micromamba refuses to create into an existing non-conda directory.
    # Treat an existing directory as an env only if it has conda metadata.
    if [ -d "$ENV_DIR/conda-meta" ]; then
      ENV_EXISTS=true
    else
      echo "Error: Prefix path '$ENV_DIR' already exists but is not a conda environment (missing conda-meta/)."
      echo "Please remove it (e.g. rm -rf '$ENV_DIR') or choose a different --prefix."
      exit 1
    fi
  else
    ENV_EXISTS=false
  fi
else
  ENV_SPEC="-n"
  ENV_LOCATION="$ENV_NAME"
  if micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    ENV_EXISTS=true
  else
    ENV_EXISTS=false
  fi
fi

# Check for C compiler (required for PyTorch compilation with torch.compile)
if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
  echo "Error: C compiler (gcc or clang) not found."
  echo "A C compiler is required for PyTorch compilation (torch.compile) to work."
  echo "Please install a C compiler:"
  echo "  Ubuntu/Debian: sudo apt-get install build-essential"
  echo "  CentOS/RHEL: sudo yum install gcc gcc-c++ make"
  echo "  macOS: brew install gcc"
  exit 1
fi

if [ "$ENV_EXISTS" = false ]; then
  if [ -n "$ENV_DIR" ]; then
    # Only ensure the parent directory exists; the prefix dir itself must not exist.
    mkdir -p "$(dirname "$ENV_DIR")"
    # Prefix mode: create the environment at the provided directory
    micromamba create --yes --prefix "$ENV_DIR" -f "$ENV_YML"
  else
    # Name mode: create using the name from environment.yml
    micromamba create --yes -f "$ENV_YML"
  fi
fi

# Install PyTorch with CUDA 12.6 FIRST (before packages that depend on it)
micromamba run "$ENV_SPEC" "$ENV_LOCATION" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install project and all dependencies (pip will see PyTorch is already installed)
# This installs: dev + docking extras (see pyproject.toml) and their deps
micromamba run "$ENV_SPEC" "$ENV_LOCATION" pip install -e ".[dev,docking]"
micromamba run "$ENV_SPEC" "$ENV_LOCATION" pre-commit install

# Verify installation
if micromamba run "$ENV_SPEC" "$ENV_LOCATION" python - <<'PY'; then
import sys

# Check package installation
try:
    import joint_improvement  # noqa: F401
    print("joint-improvement package installed")
except ImportError as e:
    print(f"Failed to import joint-improvement: {e}")
    sys.exit(1)

# Check PyTorch
try:
    import torch  # noqa: F401

    print(f"PyTorch {torch.__version__} installed")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.version.cuda}")
    else:
        print("CUDA not available (CPU-only mode)")
except ImportError as e:
    print(f"Failed to import PyTorch: {e}")
    sys.exit(1)
PY
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
