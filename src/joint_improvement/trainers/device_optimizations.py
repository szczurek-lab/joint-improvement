"""Device-specific optimizations for training."""

from __future__ import annotations

import re
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from typing import Any


def _torch_version_tuple() -> tuple[int, int, int]:
    """Best-effort parse of torch.__version__ into (major, minor, patch)."""
    v = getattr(torch, "__version__", "0.0.0")
    # Handles versions like: "2.5.1+cu124", "2.9.0a0+git....", "2.1.0"
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", str(v))
    if not m:
        return (0, 0, 0)
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def _configure_tf32_for_cuda() -> str:
    """Configure TF32 without mixing legacy/new APIs.

    Per PyTorch CUDA semantics, new fp32_precision controls were introduced (and mixing with
    legacy allow_tf32 is unsupported). We therefore:
    - Prefer the new fp32_precision controls if present (PyTorch 2.9+)
    - Fall back to legacy allow_tf32 flags otherwise

    Returns
    -------
    str
        Which API was used: "fp32_precision" or "allow_tf32".
    """
    # New API (PyTorch 2.9+ per docs): per-backend / per-op fp32_precision controls.
    has_new_api = (
        hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "fp32_precision")
        and hasattr(torch.backends, "cudnn")
        and hasattr(torch.backends.cudnn, "conv")
        and hasattr(torch.backends.cudnn.conv, "fp32_precision")
    )
    if has_new_api:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        return "fp32_precision"

    # Legacy API (pre-2.9): allow_tf32 flags.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    return "allow_tf32"


def setup_device_optimizations(
    config: Any,  # TrainerConfig - using Any to avoid circular import
    device: torch.device | str,
) -> tuple[torch.dtype, str, torch.amp.autocast | nullcontext, torch.amp.GradScaler, str]:
    """Set up device optimizations and mixed precision training.

    Returns
    -------
    tuple[torch.dtype, str, torch.amp.autocast | nullcontext, torch.amp.GradScaler, str]
        ptdtype, device_type, ctx, scaler, compile_mode
    """
    device_obj = torch.device(device) if isinstance(device, str) else device
    device_type = device_obj.type

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map[config.dtype]

    if config.dtype in ("bfloat16", "float16") and device_type != "cuda":
        raise RuntimeError(f"{config.dtype} requires CUDA")

    logger.info(f"Training with dtype: {ptdtype}")

    if device_type == "cuda":
        torch_ver = _torch_version_tuple()
        tf32_api = _configure_tf32_for_cuda()
        logger.info(f"PyTorch {torch.__version__}: TF32 configured via {tf32_api} API")
        # torch.compile exists in PyTorch 2.0+. If user asked for compilation, assert early.
        if getattr(config, "compile", False) and not hasattr(torch, "compile"):
            raise RuntimeError(
                f"TrainerConfig.compile=True requires torch.compile (PyTorch >= 2.0). "
                f"Detected torch={torch.__version__} (parsed={torch_ver})."
            )

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda" and config.dtype == "float16"))

    compile_mode = "default"

    return ptdtype, device_type, ctx, scaler, compile_mode
