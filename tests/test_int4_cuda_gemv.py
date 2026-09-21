"""Correctness tests for fused INT4 GEMV (tensor / channel / group)."""

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

GROUP_SIZES = [32, 64, 128, 256]


def _tol(k: int) -> float:
    return max(1e-1, 5e-3 * (k ** 0.5))


@requires_cuda
@pytest.mark.parametrize(
    "granularity,group_size",
    [
        ("tensor", None),
        ("channel", None),
        ("group", 32),
        ("group", 64),
        ("group", 128),
        ("group", 256),
    ],
)
@pytest.mark.parametrize(
    "n,k",
    [
        (1, 1),
        (4, 7),
        (8, 9),
        (8, 32),
        (16, 65),  # K not divisible by common group sizes
        (32, 200),
        (64, 256),
        (128, 1024),
        (256, 4096),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_int4_fused_all_granularities_vs_reference(granularity, group_size, n, k, seed):
    torch.manual_seed(seed)
    W = torch.randn(n, k) * 0.5
    x = torch.randn(k)
    qw = quantize_int4(W, granularity=granularity, group_size=group_size, pack=True)

    y_ref = gemv_from_dequant(qw, x)
    y_fused = int4_gemv_fused_from_qw(qw, x.half())
    torch.cuda.synchronize()

    assert y_fused.shape == (n,)
    err = max_abs_error(y_fused.cpu(), y_ref)
    assert err <= _tol(k), (granularity, group_size, error_summary(y_fused.cpu(), y_ref))


@requires_cuda
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("k", [100, 127, 128, 129, 255, 256, 257])
def test_int4_group_partial_last_group(group_size, k):
    torch.manual_seed(group_size + k)
    n = 8
    W = torch.randn(n, k)
    x = torch.randn(k)
    qw = quantize_int4(W, granularity="group", group_size=group_size, pack=True)
    num_groups = (k + group_size - 1) // group_size
    assert qw.scales.shape == (n, num_groups)

    y_ref = gemv_from_dequant(qw, x)
    y = int4_gemv_fused_from_qw(qw, x.half())
    torch.cuda.synchronize()
    assert max_abs_error(y.cpu(), y_ref) <= _tol(k)


@requires_cuda
@pytest.mark.parametrize("k", [1, 3, 5, 7, 15, 17, 63, 65, 127, 129])
def test_int4_fused_odd_k_all_granularities(k):
    torch.manual_seed(k)
    n = 8
    W = torch.randn(n, k)
    x = torch.randn(k)
    for granularity, group_size in [
        ("tensor", None),
        ("channel", None),
        ("group", 32),
        ("group", 128),
    ]:
        qw = quantize_int4(W, granularity=granularity, group_size=group_size, pack=True)
        y_ref = gemv_from_dequant(qw, x)
        y = int4_gemv_fused_from_qw(qw, x.half())
        torch.cuda.synchronize()
        assert max_abs_error(y.cpu(), y_ref) <= _tol(k), (granularity, group_size, k)


@requires_cuda
def test_int4_every_representable_value():
    vals = int4_representable_values()
    assert vals == list(range(INT4_QMIN, INT4_QMAX + 1))
    assert len(vals) == 16

    q = torch.tensor([vals], dtype=torch.int8)
    packed = pack_int4(q).cuda()
    x = torch.ones(16, dtype=torch.float16, device="cuda")
    y = int4_gemv_fused(packed, x, 1.0, K=16, granularity="tensor")
    torch.cuda.synchronize()
    assert y[0].item() == pytest.approx(-8.0, abs=1e-3)


@requires_cuda
def test_int4_group_known_small_example():
    # K=4, group_size=2 → 2 groups; q = [[1, 2, 3, 4]]
    q = torch.tensor([[1, 2, 3, 4]], dtype=torch.int8)
    packed = pack_int4(q).cuda()
    scales = torch.tensor([[0.5, 2.0]], dtype=torch.float32, device="cuda")
    x = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float16, device="cuda")
    y = int4_gemv_fused(
        packed, x, scales, K=4, granularity="group", group_size=2
    )
    torch.cuda.synchronize()
    # 0.5*(1+2) + 2.0*(3+4) = 1.5 + 14 = 15.5
    assert y[0].item() == pytest.approx(15.5, abs=1e-3)


@requires_cuda
def test_int4_channel_scales_layout_n():
    torch.manual_seed(0)
    n, k = 5, 64
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw = quantize_int4(W, granularity="channel", pack=True)
    scales_n = qw.scales.reshape(n).cuda()
    y = int4_gemv_fused(
        qw.qweight.cuda(), x, scales_n, K=k, granularity="channel"
    )
    y_ref = gemv_from_dequant(qw, x.cpu())
    torch.cuda.synchronize()
    assert max_abs_error(y.cpu(), y_ref) <= _tol(k)


@requires_cuda
def test_int4_fused_matches_unfused_group128():
    torch.manual_seed(0)
    n, k = 32, 257
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw = quantize_int4(W, granularity="group", group_size=128, pack=True)
    y_f = int4_gemv_fused_from_qw(qw, x)
    y_u = int4_gemv_unfused_from_qw(qw, x)
    torch.cuda.synchronize()
    assert max_abs_error(y_f.cpu(), y_u.cpu()) <= _tol(k)


@requires_cuda
def test_int4_rejects_wrong_packed_width():
    W_q = torch.zeros(2, 3, dtype=torch.uint8, device="cuda")
    x = torch.zeros(4, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="ceil\\(K/2\\)"):
        int4_gemv_fused(W_q, x, 1.0, K=4)


@requires_cuda
def test_int4_group_requires_group_size():
    W_q = torch.zeros(2, 1, dtype=torch.uint8, device="cuda")
    x = torch.zeros(2, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="group_size"):
        int4_gemv_fused(W_q, x, torch.ones(2, 1), K=2, granularity="group")
