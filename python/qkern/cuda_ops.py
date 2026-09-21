"""Python entry points for handwritten CUDA kernels."""

from __future__ import annotations

from typing import Literal

import torch

from qkern.quantization import Granularity, QuantizedWeights, dequantize, expand_scales

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
    scales: torch.Tensor | float,
    *,
    granularity: Granularity = "tensor",
    group_size: int | None = None,
) -> torch.Tensor:
    """
    Fused INT8 weight-only GEMV.

    Parameters
    ----------
    W_q:
        Contiguous CUDA ``int8`` ``[N, K]``.
    x:
        Contiguous CUDA ``float16`` ``[K]``.
    scales:
        - tensor: scalar / 0-dim / numel-1
        - channel: ``[N]`` or ``[N, 1]``
        - group: ``[N, ceil(K / group_size)]``
    granularity:
        ``"tensor"`` | ``"channel"`` | ``"group"``.
    group_size:
        Required for ``"group"`` (32 / 64 / 128 / 256 typically).
    """
    _require_ext()
    if isinstance(scales, float):
        scales_t = torch.tensor(scales, dtype=torch.float32)
    elif isinstance(scales, torch.Tensor):
        scales_t = scales
    else:
        raise TypeError("int8_gemv_fused: scales must be float or torch.Tensor")
    if granularity == "group" and group_size is None:
        raise ValueError("group_size is required when granularity='group'")
    return _C.int8_gemv_fused(W_q, x, scales_t, granularity, group_size)


def int8_gemv_unfused(
    W_q: torch.Tensor,
    x: torch.Tensor,
    scales: torch.Tensor | float,
    *,
    granularity: Granularity = "tensor",
    group_size: int | None = None,
    fp16_variant: Fp16GemvVariantName = "vec2",
) -> torch.Tensor:
    """
    Unfused INT8 path: expand scales, materialize FP16 ``Ŵ``, then FP16 GEMV.
    """
    if W_q.device.type != "cuda" or x.device.type != "cuda":
        raise ValueError("int8_gemv_unfused: W_q and x must be CUDA tensors")
    n, k = int(W_q.shape[0]), int(W_q.shape[1])
    if isinstance(scales, float):
        scales_t = torch.tensor(scales, dtype=torch.float32, device=W_q.device)
    else:
        scales_t = scales.to(device=W_q.device, dtype=torch.float32)
    scales_nk = expand_scales(
        scales_t, n=n, k=k, granularity=granularity, group_size=group_size
    )
    W_hat = (W_q.float() * scales_nk).to(dtype=torch.float16).contiguous()
    return fp16_gemv(W_hat, x.contiguous(), variant=fp16_variant)


def int8_gemv_fused_from_qw(qw: QuantizedWeights, x: torch.Tensor) -> torch.Tensor:
    """Fused GEMV from a Phase-1 ``QuantizedWeights`` (INT8, any supported granularity)."""
    if qw.bits != 8:
        raise ValueError(f"expected INT8 QuantizedWeights, got bits={qw.bits}")
    W_q = qw.qweight
    if not W_q.is_cuda:
        W_q = W_q.cuda()
    if not x.is_cuda:
        x = x.cuda()
    return int8_gemv_fused(
        W_q.contiguous(),
        x.contiguous().half(),
        qw.scales,
        granularity=qw.granularity,
        group_size=qw.group_size,
    )


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
