"""
Weight-only symmetric quantization reference (INT8 / INT4).

All routines are mathematically defined PyTorch references for later CUDA work.
They are not optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Granularity = Literal["tensor", "channel", "group"]
IntBits = Literal[4, 8]

# Signed integer ranges used by this project.
INT8_QMIN, INT8_QMAX = -128, 127
INT4_QMIN, INT4_QMAX = -8, 7


@dataclass(frozen=True)
class QuantizedWeights:
    """
    Quantized weight tensor plus scale metadata.

    Attributes
    ----------
    qweight:
        Integer tensor. For INT8: ``int8`` with shape ``[N, K]``.
        For INT4 (unpacked): ``int8`` holding values in ``[-8, 7]``, shape ``[N, K]``.
        For INT4 (packed): ``uint8`` with shape ``[N, ceil(K/2)]`` — see ``pack_int4``.
    scales:
        FP32 scales broadcastable to ``[N, K]`` after expansion:
        - tensor: shape ``[]`` (scalar)
        - channel: shape ``[N, 1]``
        - group: shape ``[N, num_groups]`` with ``num_groups = ceil(K / group_size)``
    zero_point:
        Always 0 for symmetric quantization (kept for API clarity).
    bits:
        4 or 8.
    granularity:
        ``tensor`` | ``channel`` | ``group``.
    group_size:
        Required when ``granularity == "group"``; otherwise ``None``.
    packed:
        True if INT4 values are packed two-per-byte.
    shape:
        Original weight shape ``(N, K)``.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    zero_point: int
    bits: IntBits
    granularity: Granularity
    group_size: int | None
    packed: bool
    shape: tuple[int, int]


def _qmax(bits: IntBits) -> int:
    return INT4_QMAX if bits == 4 else INT8_QMAX


def _range_for_bits(bits: IntBits) -> tuple[int, int]:
    if bits == 4:
        return INT4_QMIN, INT4_QMAX
    if bits == 8:
        return INT8_QMIN, INT8_QMAX
    raise ValueError(f"unsupported bits={bits}")


def _validate_weight(W: torch.Tensor) -> tuple[int, int]:
    if W.ndim != 2:
        raise ValueError(f"W must be 2D [N, K], got shape {tuple(W.shape)}")
    n, k = int(W.shape[0]), int(W.shape[1])
    if n < 1 or k < 1:
        raise ValueError(f"W dimensions must be positive, got {(n, k)}")
    return n, k


def _validate_granularity(granularity: Granularity, group_size: int | None, k: int) -> None:
    if granularity not in ("tensor", "channel", "group"):
        raise ValueError(f"unknown granularity={granularity!r}")
    if granularity == "group":
        if group_size is None:
            raise ValueError("group_size is required when granularity='group'")
        if int(group_size) < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        # Partial last groups are allowed when K % group_size != 0.
        _ = k
    elif group_size is not None:
        raise ValueError(f"group_size must be None for granularity={granularity!r}")


def _amax_and_scales(
    W: torch.Tensor,
    *,
    bits: IntBits,
    granularity: Granularity,
    group_size: int | None,
) -> torch.Tensor:
    """
    Compute positive FP32 scales for symmetric quantization.

    scale = max_abs / qmax
    where qmax is 127 (INT8) or 7 (INT4).

    If max_abs == 0, scale is set to 1.0 so dequantization stays well-defined.
    """
    w = W.float()
    qmax = float(_qmax(bits))
    n, k = w.shape

    if granularity == "tensor":
        max_abs = w.abs().amax()
        scale = max_abs / qmax
        if float(scale.item()) == 0.0:
            scale = torch.tensor(1.0, dtype=torch.float32, device=w.device)
        return scale.to(dtype=torch.float32)

    if granularity == "channel":
        # One scale per output row (channel).
        max_abs = w.abs().amax(dim=1, keepdim=True)  # [N, 1]
        scales = max_abs / qmax
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        return scales.to(dtype=torch.float32)

    assert granularity == "group" and group_size is not None
    gs = int(group_size)
    num_groups = (k + gs - 1) // gs
    scales = torch.empty((n, num_groups), dtype=torch.float32, device=w.device)
    for g in range(num_groups):
        start = g * gs
        end = min(start + gs, k)
        block = w[:, start:end]
        max_abs = block.abs().amax(dim=1)  # [N]
        s = max_abs / qmax
        s = torch.where(s == 0, torch.ones_like(s), s)
        scales[:, g] = s
    return scales


def expand_scales(
    scales: torch.Tensor,
    *,
    n: int,
    k: int,
    granularity: Granularity,
    group_size: int | None,
) -> torch.Tensor:
    """Expand stored scales to shape ``[N, K]`` for elementwise dequantization."""
    if granularity == "tensor":
        return scales.reshape(()).to(dtype=torch.float32).expand(n, k).clone()

    if granularity == "channel":
        if scales.shape != (n, 1) and scales.numel() != n:
            raise ValueError(f"channel scales expected [N,1] or [N], got {tuple(scales.shape)}")
        return scales.reshape(n, 1).to(dtype=torch.float32).expand(n, k).clone()

    assert granularity == "group" and group_size is not None
    gs = int(group_size)
    num_groups = (k + gs - 1) // gs
    if scales.shape != (n, num_groups):
        raise ValueError(
            f"group scales expected [{n}, {num_groups}], got {tuple(scales.shape)}"
        )
    # Repeat each group scale across up to gs columns; trim to K.
    expanded = scales.to(dtype=torch.float32).repeat_interleave(gs, dim=1)[:, :k]
    return expanded


def _quantize_symmetric(
    W: torch.Tensor,
    *,
    bits: IntBits,
    granularity: Granularity,
    group_size: int | None,
) -> QuantizedWeights:
    n, k = _validate_weight(W)
    _validate_granularity(granularity, group_size, k)
    qmin, qmax = _range_for_bits(bits)

    scales = _amax_and_scales(W, bits=bits, granularity=granularity, group_size=group_size)
    scales_nk = expand_scales(
        scales, n=n, k=k, granularity=granularity, group_size=group_size
    )

    # q = clip(round(W / scale), qmin, qmax)
    q = torch.round(W.float() / scales_nk)
    q = torch.clamp(q, qmin, qmax).to(torch.int8)

    return QuantizedWeights(
        qweight=q,
        scales=scales,
        zero_point=0,
        bits=bits,
        granularity=granularity,
        group_size=int(group_size) if group_size is not None else None,
        packed=False,
        shape=(n, k),
    )


def quantize_int8(
    W: torch.Tensor,
    *,
    granularity: Granularity = "tensor",
    group_size: int | None = None,
) -> QuantizedWeights:
    """
    Symmetric weight-only INT8 quantization.

    Parameters
    ----------
    W:
        FP weight matrix ``[N, K]``.
    granularity:
        ``"tensor"``, ``"channel"``, or ``"group"``.
    group_size:
        Required for ``granularity="group"``. Last group may be shorter than
        ``group_size`` when ``K`` is not divisible by ``group_size``.
    """
    return _quantize_symmetric(W, bits=8, granularity=granularity, group_size=group_size)


def quantize_int4(
    W: torch.Tensor,
    *,
    granularity: Granularity = "tensor",
    group_size: int | None = None,
    pack: bool = False,
) -> QuantizedWeights:
    """
    Symmetric weight-only signed INT4 quantization with values in ``[-8, 7]``.

    If ``pack=True``, ``qweight`` is uint8 packed storage (two INT4 per byte).
    """
    q = _quantize_symmetric(W, bits=4, granularity=granularity, group_size=group_size)
    if not pack:
        return q
    packed = pack_int4(q.qweight)
    return QuantizedWeights(
        qweight=packed,
        scales=q.scales,
        zero_point=0,
        bits=4,
        granularity=q.granularity,
        group_size=q.group_size,
        packed=True,
        shape=q.shape,
    )


def dequantize(
    qw: QuantizedWeights,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize to a dense floating-point matrix of shape ``[N, K]``.

    ``W_hat = q * scale`` (zero-point is 0).
    """
    n, k = qw.shape
    if qw.bits == 4 and qw.packed:
        q = unpack_int4(qw.qweight, k=k)
    else:
        q = qw.qweight
        if q.shape != (n, k):
            raise ValueError(f"qweight shape {tuple(q.shape)} != {(n, k)}")

    scales_nk = expand_scales(
        qw.scales,
        n=n,
        k=k,
        granularity=qw.granularity,
        group_size=qw.group_size,
    )
    return (q.float() * scales_nk).to(dtype=dtype)


def dequantize_int8(qw: QuantizedWeights, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if qw.bits != 8:
        raise ValueError(f"expected INT8 QuantizedWeights, got bits={qw.bits}")
    return dequantize(qw, dtype=dtype)


def dequantize_int4(qw: QuantizedWeights, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if qw.bits != 4:
        raise ValueError(f"expected INT4 QuantizedWeights, got bits={qw.bits}")
    return dequantize(qw, dtype=dtype)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """
    Pack signed INT4 values in ``[-8, 7]`` into uint8 bytes (two values per byte).

    Layout (little-endian nibbles within each byte):
      - low nibble  = q[..., 2*i]
      - high nibble = q[..., 2*i + 1]   (0 if K is odd and this is padding)

    Signed values are stored as 4-bit two's-complement nibbles:
      stored = value & 0xF
    so ``-8 -> 0x8``, ``-1 -> 0xF``, ``0 -> 0x0``, ``7 -> 0x7``.

    Input shape ``[..., K]`` -> output shape ``[..., ceil(K/2)]``.
    """
    if q.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"pack_int4 expects integer q, got {q.dtype}")
    qmin, qmax = INT4_QMIN, INT4_QMAX
    if bool(((q < qmin) | (q > qmax)).any().item()):
        raise ValueError(f"INT4 values must be in [{qmin}, {qmax}]")

    k = q.shape[-1]
    # Pad odd K with a zero nibble so packing is uniform.
    if k % 2 == 1:
        pad = torch.zeros(*q.shape[:-1], 1, dtype=torch.int8, device=q.device)
        q = torch.cat([q.to(torch.int8), pad], dim=-1)

    lo = q[..., 0::2].to(torch.int16) & 0xF
    hi = q[..., 1::2].to(torch.int16) & 0xF
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed


def unpack_int4(packed: torch.Tensor, *, k: int) -> torch.Tensor:
    """
    Unpack uint8 packed INT4 storage back to signed ``int8`` values in ``[-8, 7]``.

    Parameters
    ----------
    packed:
        Shape ``[..., ceil(K/2)]``, dtype uint8 (or integer byte values).
    k:
        Original number of INT4 elements along the last dimension.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    expected = (k + 1) // 2
    if packed.shape[-1] != expected:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} != ceil(k/2)={expected} for k={k}"
        )

    p = packed.to(torch.int16)
    lo = p & 0xF
    hi = (p >> 4) & 0xF

    def _sign_extend_nibble(nibble: torch.Tensor) -> torch.Tensor:
        # Values 8..15 represent -8..-1.
        return torch.where(nibble >= 8, nibble - 16, nibble).to(torch.int8)

    even = _sign_extend_nibble(lo)
    odd = _sign_extend_nibble(hi)
    # Interleave: [e0, o0, e1, o1, ...]
    out = torch.stack((even, odd), dim=-1).reshape(*packed.shape[:-1], expected * 2)
    return out[..., :k]


def int4_representable_values() -> list[int]:
    """All signed INT4 values used by this project: ``[-8, -7, ..., 7]``."""
    return list(range(INT4_QMIN, INT4_QMAX + 1))
