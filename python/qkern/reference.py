"""PyTorch reference implementations for QKern (correctness baselines)."""

from __future__ import annotations

import torch

from qkern.quantization import (
    Granularity,
    QuantizedWeights,
    dequantize,
    quantize_int4,
    quantize_int8,
)


def fp16_gemv(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reference FP16 GEMV: ``y = W @ x``.

    Parameters
    ----------
    W:
        Weight matrix ``[N, K]`` (any floating dtype; cast to FP16 for the product).
    x:
        Activation vector ``[K]``.

    Returns
    -------
    y:
        Output vector ``[N]`` in FP32.

    Notes
    -----
    Products are computed in FP16 then accumulated in FP32 so this routine is a
    stable correctness reference for later CUDA kernels (which typically use
    FP32 accumulators as well).
    """
    if W.ndim != 2:
        raise ValueError(f"W must be [N, K], got shape {tuple(W.shape)}")
    if x.ndim != 1:
        raise ValueError(f"x must be [K], got shape {tuple(x.shape)}")
    n, k = W.shape
    if x.shape[0] != k:
        raise ValueError(f"x.shape[0]={x.shape[0]} != K={k}")

    w16 = W.to(dtype=torch.float16)
    x16 = x.to(dtype=torch.float16)
    # Explicit FP32 accumulation: sum_k float32(float16(W[n,k]) * float16(x[k])).
    y = (w16.float() * x16.float().unsqueeze(0)).sum(dim=1)
    assert y.shape == (n,)
    return y


def gemv_from_dequant(
    qw: QuantizedWeights,
    x: torch.Tensor,
    *,
    weight_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Reference quantized GEMV: dequantize weights, then call ``fp16_gemv``.

    This is the unfused reference path (dequant then matvec), used to validate
    quantization math before fused CUDA kernels exist.
    """
    W_hat = dequantize(qw, dtype=weight_dtype)
    return fp16_gemv(W_hat, x)


def int8_gemv_reference(
    W: torch.Tensor,
    x: torch.Tensor,
    *,
    granularity: Granularity = "tensor",
    group_size: int | None = None,
) -> tuple[torch.Tensor, QuantizedWeights]:
    """Quantize ``W`` to INT8, dequantize, then GEMV. Returns ``(y, qw)``."""
    qw = quantize_int8(W, granularity=granularity, group_size=group_size)
    y = gemv_from_dequant(qw, x)
    return y, qw


def int4_gemv_reference(
    W: torch.Tensor,
    x: torch.Tensor,
    *,
    granularity: Granularity = "group",
    group_size: int | None = 128,
    pack: bool = True,
) -> tuple[torch.Tensor, QuantizedWeights]:
    """Quantize ``W`` to INT4 (optionally packed), dequantize, then GEMV."""
    qw = quantize_int4(
        W,
        granularity=granularity,
        group_size=group_size,
        pack=pack,
    )
    y = gemv_from_dequant(qw, x)
    return y, qw
