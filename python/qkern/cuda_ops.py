"""Python entry points for handwritten CUDA kernels."""

from __future__ import annotations

from typing import Literal

import torch

try:
    from . import _C
except ImportError:  # pragma: no cover
    _C = None  # type: ignore[assignment]

Fp16GemvVariantName = Literal["naive", "x_smem"]


def is_cuda_extension_available() -> bool:
    return _C is not None


def fp16_gemv(
    W: torch.Tensor,
    x: torch.Tensor,
    *,
    variant: Fp16GemvVariantName = "naive",
) -> torch.Tensor:
    """
    CUDA FP16 GEMV: ``y = W @ x`` (FP32 accumulation).

    Parameters
    ----------
    W, x:
        Contiguous CUDA ``float16`` tensors ``[N, K]`` and ``[K]``.
    variant:
        ``"naive"`` — Phase-2 baseline (one thread per row, global ``x`` loads).
        ``"x_smem"`` — same mapping with ``x`` tiled through shared memory.
    """
    if _C is None:
        raise ImportError(
            "qkern CUDA extension is not built. Install with: python setup.py build_ext --inplace"
        )
    if not isinstance(W, torch.Tensor) or not isinstance(x, torch.Tensor):
        raise TypeError("fp16_gemv: W and x must be torch.Tensor")
    return _C.fp16_gemv(W, x, variant)


def fp16_gemv_naive(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Phase-2 baseline kernel."""
    return fp16_gemv(W, x, variant="naive")


def fp16_gemv_x_smem(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Activation-caching optimization (shared-memory tiles of ``x``)."""
    return fp16_gemv(W, x, variant="x_smem")
