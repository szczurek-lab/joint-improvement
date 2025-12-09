"""Device-specific optimizations for training.

This module provides utilities for setting up device optimizations including
CUDA settings, mixed precision training, and GPU-specific optimizations (e.g., H100).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from typing import Any


def setup_device_optimizations(
    config: Any,  # TrainerConfig - using Any to avoid circular import
    device: torch.device | str,
) -> tuple[torch.dtype, str, torch.amp.autocast | nullcontext, torch.amp.GradScaler, str]:
    """Set up device-specific optimizations and mixed precision training.

    Configures CUDA optimizations (TF32, H100-specific settings), validates
    dtype compatibility, and sets up autocast context and gradient scaler.

    Parameters
    ----------
    config
        Training configuration (TrainerConfig). Must have a `dtype` attribute.
    device : torch.device | str
        Device to train on.

    Returns
    -------
    tuple[torch.dtype, str, torch.amp.autocast | nullcontext, torch.amp.GradScaler, str]
        Tuple containing:
        - ptdtype: PyTorch dtype for mixed precision
        - device_type: Device type string ("cuda" or "cpu")
        - ctx: Autocast context manager
        - scaler: Gradient scaler (enabled only for float16)
        - compile_mode: Recommended torch.compile mode

    Examples
    --------
    >>> from joint_improvement.trainers.multitask import TrainerConfig
    >>> from joint_improvement.trainers.device_optimizations import setup_device_optimizations
    >>> config = TrainerConfig(dtype="bfloat16")
    >>> ptdtype, device_type, ctx, scaler, compile_mode = setup_device_optimizations(config, device="cuda")
    """
    # Step 1: Determine actual device type
    device_obj = torch.device(device) if isinstance(device, str) else device
    is_cuda_available = torch.cuda.is_available()
    is_cuda_device = device_obj.type == "cuda"

    if is_cuda_device and not is_cuda_available:
        raise RuntimeError(
            f"CUDA device requested ({device_obj}) but CUDA is not available. "
            "Please use CPU device or ensure CUDA is properly installed."
        )

    device_type = device_obj.type

    # Step 2: Detect GPU architecture for optimizations
    is_h100 = False
    gpu_name = None
    if device_type == "cuda" and torch.cuda.device_count() > 0:
        try:
            gpu_name = torch.cuda.get_device_name(device_obj.index if device_obj.index is not None else 0)
            is_h100 = "H100" in gpu_name or "Hopper" in gpu_name
            logger.info(f"GPU detected: {gpu_name}")
        except Exception as e:
            logger.warning(f"Could not detect GPU name: {e}")

    # Step 3: Validate and set dtype
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    requested_dtype = config.dtype
    ptdtype = dtype_map[requested_dtype]

    # Validate dtype compatibility with device
    if requested_dtype == "bfloat16" and device_type != "cuda":
        raise RuntimeError(
            f"bfloat16 requested but device is {device_type}. "
            "bfloat16 requires CUDA. Please use CUDA device or change dtype to float32."
        )

    if requested_dtype == "float16" and device_type != "cuda":
        raise RuntimeError(
            f"float16 requested but device is {device_type}. "
            "float16 requires CUDA. Please use CUDA device or change dtype to float32."
        )

    logger.info(f"Training with dtype: {ptdtype}")

    # Step 4: Apply CUDA optimizations
    if device_type == "cuda":
        logger.info("Applying CUDA optimizations...")

        # Enable TF32 for Ampere+ GPUs (A100, H100, RTX 30xx, etc.)
        # Use new API if available, fallback to deprecated API for compatibility
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            logger.info("Enabled TF32 for CUDA matrix multiplications")
        elif hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("Enabled TF32 for CUDA matrix multiplications")

        if hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            logger.info("Enabled TF32 for cuDNN operations")
        elif hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for cuDNN operations")

    # Step 5: Determine compile mode
    if is_h100:
        compile_mode = "max-autotune"
    else:
        compile_mode = "reduce-overhead"
    logger.info(f"Compiling with mode: {compile_mode}")

    # Step 6: Set up autocast context
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Step 7: Set up gradient scaler
    # Only needed for float16 (bfloat16 has better numerical stability)
    use_scaler = device_type == "cuda" and requested_dtype == "float16"
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    return ptdtype, device_type, ctx, scaler, compile_mode
