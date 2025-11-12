# Joint Improvement

Official repository for Joint Improvement.

## Environment Setup (Micromamba)

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) and run the helper script:

   ```bash
   scripts/setup_env.sh
   ```

The setup script will:
1. Create a conda environment with Python 3.12
2. Install PyTorch with CUDA 12.6 support
3. Install the project package and all dependencies (including dev dependencies)
4. Set up pre-commit hooks

**Note**: PyTorch is installed first with CUDA 12.6 support before other dependencies to ensure compatibility. For CPU-only setups, modify `scripts/setup_env.sh` to install PyTorch without the CUDA index URL.
