"""Python entry points for handwritten CUDA kernels."""

from __future__ import annotations

from typing import Literal

import torch

from qkern.quantization import QuantizedWeights, dequantize

try:
    from . import _C
except ImportError:  # pragma: no cover
    _C = None  # type: ignore[assignment]

Fp16GemvVariantName = Literal["naive", "x_smem", "vec2"]


def is_cuda_extension_available() -> bool:
    return _C is not None


def _require_ext() -> None:
    if _C is None:
        raise ImportError(
            "qkern CUDA extension is not built. Install with: python setup.py build_ext --inplace"
        )


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
        ``"naive"`` — Phase-2 baseline.
        ``"x_smem"`` — shared-memory tiles of ``x``.
        ``"vec2"`` — safe ``__half2`` loads with scalar fallback.
    """
    _require_ext()
    if not isinstance(W, torch.Tensor) or not isinstance(x, torch.Tensor):
        raise TypeError("fp16_gemv: W and x must be torch.Tensor")
    return _C.fp16_gemv(W, x, variant)


def fp16_gemv_naive(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Phase-2 baseline kernel."""
    return fp16_gemv(W, x, variant="naive")


def fp16_gemv_x_smem(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Activation-caching optimization (shared-memory tiles of ``x``)."""
    return fp16_gemv(W, x, variant="x_smem")


def fp16_gemv_vec2(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Safe ``__half2`` vectorized loads with scalar fallback."""
    return fp16_gemv(W, x, variant="vec2")


def int8_gemv_fused(
    W_q: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """
    Fused INT8 weight-only GEMV (per-tensor scale).

    Computes ``y[n] = scale * sum_k float(W_q[n,k]) * float(x[k])`` on device
    without writing a dequantized ``[N, K]`` weight matrix to global memory.

    Parameters
    ----------
    W_q:
        Contiguous CUDA ``int8`` tensor ``[N, K]``.
    x:
        Contiguous CUDA ``float16`` tensor ``[K]``.
    scale:
        Per-tensor positive scale (Python float or 0-dim / 1-element tensor).
    """
    _require_ext()
    if isinstance(scale, float):
        scale_t = torch.tensor(scale, dtype=torch.float32)
    elif isinstance(scale, torch.Tensor):
        scale_t = scale
    else:
        raise TypeError("int8_gemv_fused: scale must be float or torch.Tensor")
    return _C.int8_gemv_fused(W_q, x, scale_t)


def int8_gemv_unfused(
    W_q: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    fp16_variant: Fp16GemvVariantName = "vec2",
) -> torch.Tensor:
    """
    Unfused INT8 path: materialize FP16 dequantized weights, then FP16 GEMV.

    ``W_hat = float(W_q) * scale`` (cast to FP16), then ``fp16_gemv(W_hat, x)``.
    This intentionally writes a full ``[N, K]`` FP16 tensor — the fusion baseline.
    """
    if isinstance(scale, float):
        scale_f = scale
    elif isinstance(scale, torch.Tensor):
        scale_f = float(scale.detach().float().reshape(-1)[0].item())
    else:
        raise TypeError("int8_gemv_unfused: scale must be float or torch.Tensor")

    if W_q.device.type != "cuda" or x.device.type != "cuda":
        raise ValueError("int8_gemv_unfused: W_q and x must be CUDA tensors")

    # Materialize dequantized weights in global memory (the cost fusion avoids).
    W_hat = (W_q.float() * scale_f).to(dtype=torch.float16).contiguous()
    return fp16_gemv(W_hat, x.contiguous(), variant=fp16_variant)


def int8_gemv_fused_from_qw(qw: QuantizedWeights, x: torch.Tensor) -> torch.Tensor:
    """Convenience: fused GEMV from a per-tensor ``QuantizedWeights`` object."""
    if qw.bits != 8:
        raise ValueError(f"expected INT8 QuantizedWeights, got bits={qw.bits}")
    if qw.granularity != "tensor":
        raise ValueError(
            f"fused INT8 kernel currently supports per-tensor only, got {qw.granularity}"
        )
    W_q = qw.qweight
    if not W_q.is_cuda:
        W_q = W_q.cuda()
    if not x.is_cuda:
        x = x.cuda()
    return int8_gemv_fused(W_q.contiguous(), x.contiguous().half(), qw.scales)


def int8_gemv_unfused_from_qw(
    qw: QuantizedWeights,
    x: torch.Tensor,
    *,
    fp16_variant: Fp16GemvVariantName = "vec2",
) -> torch.Tensor:
    """Unfused path via explicit ``dequantize`` then CUDA FP16 GEMV."""
    if qw.bits != 8:
        raise ValueError(f"expected INT8 QuantizedWeights, got bits={qw.bits}")
    W_hat = dequantize(qw, dtype=torch.float16)
    if not W_hat.is_cuda:
        W_hat = W_hat.cuda()
    if not x.is_cuda:
        x = x.cuda()
    return fp16_gemv(W_hat.contiguous(), x.contiguous().half(), variant=fp16_variant)
