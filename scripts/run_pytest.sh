#!/bin/bash
# Wrapper script for pytest that uses the joint-improvement conda environment

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Path to the conda environment
ENV_PATH="$PROJECT_ROOT/Y/envs/joint-improvement"
PYTHON="$ENV_PATH/bin/python"

# Check if the environment exists
if [ ! -f "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON" >&2
    echo "Please ensure the joint-improvement conda environment is set up." >&2
    exit 1
fi

# Print debug information about which environment is being used
echo "Using Python: $PYTHON" >&2
echo "Python version: $($PYTHON --version 2>&1)" >&2
echo "Pytest version: $($PYTHON -m pytest --version 2>&1)" >&2
echo "Environment path: $ENV_PATH" >&2

# Run pytest with the environment's Python
export PYTHONPATH="$PROJECT_ROOT/src"
exec "$PYTHON" -m pytest "$PROJECT_ROOT/tests/" -x --tb=short -p no:xdist "$@"

