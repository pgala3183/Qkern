"""Correctness tests for fused INT4 GEMV (per-tensor, packed)."""

from __future__ import annotations

import pytest
import torch

from qkern import (
    INT4_QMAX,
    INT4_QMIN,
    error_summary,
    gemv_from_dequant,
    int4_gemv_fused,
    int4_gemv_fused_from_qw,
    int4_gemv_unfused_from_qw,
    int4_representable_values,
    is_cuda_extension_available,
    max_abs_error,
    pack_int4,
    quantize_int4,
)

requires_cuda = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not is_cuda_extension_available()),
    reason="CUDA device + qkern._C extension required",
)


def _tol(k: int) -> float:
    return max(1e-1, 5e-3 * (k ** 0.5))


@requires_cuda
@pytest.mark.parametrize(
    "n,k",
    [
        (1, 1),
        (1, 2),
        (4, 7),  # odd K
        (8, 9),  # odd K
        (8, 32),
        (16, 65),
        (32, 200),
        (64, 256),
        (128, 1024),
        (256, 4096),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_int4_fused_vs_reference_random(n, k, seed):
    torch.manual_seed(seed)
    W = torch.randn(n, k) * 0.5
    x = torch.randn(k)
    qw = quantize_int4(W, granularity="tensor", pack=True)

    y_ref = gemv_from_dequant(qw, x)
    y_fused = int4_gemv_fused_from_qw(qw, x.half())
    torch.cuda.synchronize()

    assert y_fused.shape == (n,)
    err = max_abs_error(y_fused.cpu(), y_ref)
    assert err <= _tol(k), error_summary(y_fused.cpu(), y_ref)


@requires_cuda
@pytest.mark.parametrize("k", [1, 3, 5, 7, 15, 17, 63, 65, 127, 129])
def test_int4_fused_odd_k(k):
    torch.manual_seed(k)
    n = 8
    W = torch.randn(n, k)
    x = torch.randn(k)
    qw = quantize_int4(W, granularity="tensor", pack=True)
    assert qw.qweight.shape[1] == (k + 1) // 2

    y_ref = gemv_from_dequant(qw, x)
    y = int4_gemv_fused_from_qw(qw, x.half())
    torch.cuda.synchronize()
    assert max_abs_error(y.cpu(), y_ref) <= _tol(k)


@requires_cuda
def test_int4_every_representable_value():
    """Every signed INT4 value appears as a weight; compare to reference."""
    vals = int4_representable_values()
    assert vals == list(range(INT4_QMIN, INT4_QMAX + 1))
    assert len(vals) == 16

    # One row with all 16 values (K=16, even); scale=1 so dequant == q.
    q = torch.tensor([vals], dtype=torch.int8)
    packed = pack_int4(q).cuda()
    x = torch.ones(16, dtype=torch.float16, device="cuda")
    y = int4_gemv_fused(packed, x, 1.0, K=16, granularity="tensor")
    torch.cuda.synchronize()
    # Σ q[k] * 1 = sum(vals) = (-8)+(-7)+...+7 = -8
    assert y[0].item() == pytest.approx(-8.0, abs=1e-3)


@requires_cuda
def test_int4_every_value_as_pair_pack():
    """Exhaustive low/high nibble pairs via Phase-1 packing."""
    vals = int4_representable_values()
    rows = []
    for lo in vals:
        for hi in vals:
            rows.append([lo, hi])
    q = torch.tensor(rows, dtype=torch.int8)  # [256, 2]
    packed = pack_int4(q).cuda()
    x = torch.tensor([2.0, 3.0], dtype=torch.float16, device="cuda")
    scale = 0.5
    y = int4_gemv_fused(packed, x, scale, K=2)
    torch.cuda.synchronize()

    expected = (q.float() * scale) @ x.cpu().float()
    assert max_abs_error(y.cpu(), expected) <= 1e-3


@requires_cuda
def test_int4_fused_known_small_example():
    # q = [[-8, 7], [1, -1]] → bytes [0x78, 0xF1]
    q = torch.tensor([[-8, 7], [1, -1]], dtype=torch.int8)
    packed = pack_int4(q).cuda()
    assert packed.tolist() == [[0x78], [0xF1]]
    x = torch.tensor([2.0, 4.0], dtype=torch.float16, device="cuda")
    y = int4_gemv_fused(packed, x, 0.5, K=2)
    torch.cuda.synchronize()
    # row0: 0.5 * (-8*2 + 7*4) = 0.5 * 12 = 6
    # row1: 0.5 * (1*2 + (-1)*4) = 0.5 * (-2) = -1
    assert y[0].item() == pytest.approx(6.0, abs=1e-3)
    assert y[1].item() == pytest.approx(-1.0, abs=1e-3)


@requires_cuda
def test_int4_fused_matches_unfused():
    torch.manual_seed(0)
    n, k = 32, 257  # odd K
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw = quantize_int4(W, granularity="tensor", pack=True)
    y_f = int4_gemv_fused_from_qw(qw, x)
    y_u = int4_gemv_unfused_from_qw(qw, x)
    torch.cuda.synchronize()
    assert max_abs_error(y_f.cpu(), y_u.cpu()) <= _tol(k)


@requires_cuda
def test_int4_rejects_wrong_packed_width():
    W_q = torch.zeros(2, 3, dtype=torch.uint8, device="cuda")  # claims ceil(K/2)=3 → K=5 or 6
    x = torch.zeros(4, dtype=torch.float16, device="cuda")  # K=4 → ceil=2
    with pytest.raises(RuntimeError, match="ceil\\(K/2\\)"):
        int4_gemv_fused(W_q, x, 1.0, K=4)


@requires_cuda
def test_int4_rejects_non_tensor_granularity():
    W_q = torch.zeros(2, 1, dtype=torch.uint8, device="cuda")
    x = torch.zeros(2, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="tensor"):
        int4_gemv_fused(W_q, x, 1.0, K=2, granularity="channel")
