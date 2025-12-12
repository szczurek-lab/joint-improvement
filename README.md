# Joint Improvement

Official repository for Joint Improvement.

## Installation

Install [mamba](https://mamba.readthedocs.io/en/latest/installation.html) (or micromamba) and run:

   ```bash
   # Using mamba
   ./scripts/setup_env_mamba.sh
   ./scripts/setup_env_mamba.sh --prefix /custom/path/to/env

   # Using micromamba 
   ./scripts/setup_env_micromamba.sh
   ./scripts/setup_env_micromamba.sh --prefix /custom/path/to/env
   ```

The setup script will:
1. Check for C compiler (gcc/clang) - exits with error if not found (required for `torch.compile`)
2. Create a conda environment with Python 3.10
3. Install PyTorch with CUDA 12.6 support
4. Install the project package and all dependencies (including dev dependencies)
5. Set up pre-commit hooks

**Note**: PyTorch is installed first with CUDA 12.6 support before other dependencies to ensure compatibility. For CPU-only setups, modify the setup script to install PyTorch without the CUDA index URL.

## Pre-commit / pytest environment (Cursor-friendly)

The `pytest` pre-commit hook runs via `mamba`/`micromamba` and requires an explicit **environment prefix**.

```bash
git config joint-improvement.envPrefix /absolute/path/to/your/env
```
