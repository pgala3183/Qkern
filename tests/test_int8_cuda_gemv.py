"""Correctness tests for fused INT8 GEMV (tensor / channel / group)."""

from __future__ import annotations

import pytest
import torch

from qkern import (
    error_summary,
    gemv_from_dequant,
    int8_gemv_fused,
    int8_gemv_fused_from_qw,
    int8_gemv_unfused_from_qw,
    is_cuda_extension_available,
    max_abs_error,
    quantize_int8,
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
        (8, 32),
        (16, 65),  # K not divisible by common group sizes
        (32, 200),
        (64, 256),
        (128, 1024),
        (256, 4096),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_int8_fused_all_granularities_vs_reference(granularity, group_size, n, k, seed):
    if granularity == "group" and group_size is not None and k < 1:
        pytest.skip("empty")
    torch.manual_seed(seed)
    W = torch.randn(n, k) * 0.5
    x = torch.randn(k)
    qw = quantize_int8(W, granularity=granularity, group_size=group_size)

    y_ref = gemv_from_dequant(qw, x)
    y_fused = int8_gemv_fused_from_qw(qw, x.half())
    torch.cuda.synchronize()

    assert y_fused.shape == (n,)
    err = max_abs_error(y_fused.cpu(), y_ref)
    assert err <= _tol(k), (granularity, group_size, error_summary(y_fused.cpu(), y_ref))


@requires_cuda
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("k", [100, 127, 128, 129, 255, 256, 257])
def test_int8_group_partial_last_group(group_size, k):
    torch.manual_seed(group_size + k)
    n = 8
    W = torch.randn(n, k)
    x = torch.randn(k)
    qw = quantize_int8(W, granularity="group", group_size=group_size)
    num_groups = (k + group_size - 1) // group_size
    assert qw.scales.shape == (n, num_groups)

    y_ref = gemv_from_dequant(qw, x)
    y = int8_gemv_fused_from_qw(qw, x.half())
    torch.cuda.synchronize()
    assert max_abs_error(y.cpu(), y_ref) <= _tol(k)


@requires_cuda
def test_int8_channel_scales_layout_n():
    torch.manual_seed(0)
    n, k = 5, 64
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw = quantize_int8(W, granularity="channel")
    # Accept [N,1] from quantize; also exercise flat [N].
    scales_n = qw.scales.reshape(n).cuda()
    y = int8_gemv_fused(
        qw.qweight.cuda(), x, scales_n, granularity="channel"
    )
    y_ref = gemv_from_dequant(qw, x.cpu())
    torch.cuda.synchronize()
    assert max_abs_error(y.cpu(), y_ref) <= _tol(k)


@requires_cuda
def test_int8_fused_known_small_example():
    W_q = torch.tensor([[1, -2], [3, 0]], dtype=torch.int8, device="cuda")
    x = torch.tensor([2.0, 4.0], dtype=torch.float16, device="cuda")
    y = int8_gemv_fused(W_q, x, 0.5, granularity="tensor")
    torch.cuda.synchronize()
    assert y[0].item() == pytest.approx(-3.0, abs=1e-3)
    assert y[1].item() == pytest.approx(3.0, abs=1e-3)


@requires_cuda
def test_int8_channel_known_example():
    # scales differ per row
    W_q = torch.tensor([[2, 2], [2, 2]], dtype=torch.int8, device="cuda")
    x = torch.tensor([1.0, 1.0], dtype=torch.float16, device="cuda")
    scales = torch.tensor([0.5, 2.0], dtype=torch.float32, device="cuda")
    y = int8_gemv_fused(W_q, x, scales, granularity="channel")
    torch.cuda.synchronize()
    # y0 = 0.5*(2+2)=2, y1 = 2*(2+2)=8
    assert y[0].item() == pytest.approx(2.0, abs=1e-3)
    assert y[1].item() == pytest.approx(8.0, abs=1e-3)


@requires_cuda
def test_int8_group_known_example():
    # K=4, gs=2 → 2 groups. W_q row0 = [1,1,2,2], scales=[1.0, 0.5]
    W_q = torch.tensor([[1, 1, 2, 2]], dtype=torch.int8, device="cuda")
    x = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float16, device="cuda")
    scales = torch.tensor([[1.0, 0.5]], dtype=torch.float32, device="cuda")
    y = int8_gemv_fused(W_q, x, scales, granularity="group", group_size=2)
    torch.cuda.synchronize()
    # 1*1*1 + 1*1*1 + 2*0.5*1 + 2*0.5*1 = 1+1+1+1 = 4
    assert y[0].item() == pytest.approx(4.0, abs=1e-3)


@requires_cuda
def test_int8_rejects_wrong_dtype():
    W_q = torch.randn(4, 8, device="cuda").to(torch.float16)
    x = torch.randn(8, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="int8"):
        int8_gemv_fused(W_q, x, 1.0)


@requires_cuda
def test_int8_rejects_cpu():
    W_q = torch.randint(-10, 10, (4, 8), dtype=torch.int8)
    x = torch.randn(8, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="CUDA"):
        int8_gemv_fused(W_q, x, 1.0)


@requires_cuda
def test_int8_rejects_bad_tensor_scale():
    W_q = torch.randint(-10, 10, (4, 8), dtype=torch.int8, device="cuda")
    x = torch.randn(8, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="numel"):
        int8_gemv_fused(W_q, x, torch.tensor([1.0, 2.0]), granularity="tensor")


@requires_cuda
def test_int8_group_requires_group_size():
    W_q = torch.randint(-10, 10, (4, 8), dtype=torch.int8, device="cuda")
    x = torch.randn(8, dtype=torch.float16, device="cuda")
    scales = torch.randn(4, 1, device="cuda")
    with pytest.raises(ValueError, match="group_size"):
        int8_gemv_fused(W_q, x, scales, granularity="group", group_size=None)


@requires_cuda
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_unfused_matches_reference_group(group_size):
    torch.manual_seed(9)
    W = torch.randn(16, 200)
    x = torch.randn(200)
    qw = quantize_int8(W, granularity="group", group_size=group_size)
    y_ref = gemv_from_dequant(qw, x)
    y_u = int8_gemv_unfused_from_qw(qw, x.half())
    torch.cuda.synchronize()
    assert max_abs_error(y_u.cpu(), y_ref) <= _tol(200)
