# Joint Improvement

Official repository for Joint Improvement.

## Environment Setup (Micromamba)

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) and run the helper script:

   ```bash
   scripts/setup_env.sh
   ```

The setup script will:
1. Check for and install C compiler (gcc/clang) if needed (required for `torch.compile`)
2. Create a conda environment with Python 3.12
3. Install PyTorch with CUDA 12.6 support
4. Install the project package and all dependencies (including dev dependencies)
5. Set up pre-commit hooks

**Note**: PyTorch is installed first with CUDA 12.6 support before other dependencies to ensure compatibility. For CPU-only setups, modify `scripts/setup_env.sh` to install PyTorch without the CUDA index URL.

**C Compiler Requirement**: A C compiler (gcc or clang) is required for PyTorch compilation (`torch.compile`) to work. The setup script will attempt to install `build-essential` on Ubuntu/Debian systems automatically. On other systems, install gcc/clang manually.
