"""Correctness tests for fused INT8 GEMV (per-tensor) vs Phase 1 reference."""

from __future__ import annotations

import pytest
import torch

from qkern import (
    error_summary,
    fp16_gemv_ref,
    gemv_from_dequant,
    int8_gemv_fused,
    int8_gemv_fused_from_qw,
    int8_gemv_unfused,
    int8_gemv_unfused_from_qw,
    is_cuda_extension_available,
    max_abs_error,
    quantize_int8,
)

requires_cuda = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not is_cuda_extension_available()),
    reason="CUDA device + qkern._C extension required",
)


def _tol(k: int) -> float:
    # INT8 dequant + FP16 x + FP32 accum; allow slack that grows slowly with K.
    return max(1e-1, 5e-3 * (k ** 0.5))


@requires_cuda
@pytest.mark.parametrize(
    "n,k",
    [
        (1, 1),
        (4, 7),
        (8, 32),
        (16, 65),
        (64, 256),
        (128, 1024),
        (256, 4096),
        (1024, 4096),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_int8_fused_matches_phase1_reference(n, k, seed):
    torch.manual_seed(seed)
    W = torch.randn(n, k) * 0.5
    x = torch.randn(k)
    qw = quantize_int8(W, granularity="tensor")

    y_ref = gemv_from_dequant(qw, x)  # Phase-1 unfused PyTorch reference
    y_fused = int8_gemv_fused_from_qw(qw, x.half())
    torch.cuda.synchronize()

    assert y_fused.dtype == torch.float32
    assert y_fused.shape == (n,)
    err = max_abs_error(y_fused.cpu(), y_ref)
    assert err <= _tol(k), error_summary(y_fused.cpu(), y_ref)


@requires_cuda
@pytest.mark.parametrize("n,k", [(8, 33), (32, 128), (64, 4096)])
@pytest.mark.parametrize("seed", [0, 7])
def test_int8_fused_matches_unfused_cuda(n, k, seed):
    torch.manual_seed(seed)
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw = quantize_int8(W, granularity="tensor")
    W_q = qw.qweight.cuda().contiguous()
    scale = float(qw.scales.item())

    y_f = int8_gemv_fused(W_q, x, scale)
    y_u = int8_gemv_unfused(W_q, x, scale)
    torch.cuda.synchronize()
    assert max_abs_error(y_f, y_u) <= _tol(k)


@requires_cuda
def test_int8_fused_known_small_example():
    # W_q = [[1, -2], [3, 0]], scale=0.5, x=[2, 4]
    # y0 = 0.5*(1*2 + (-2)*4) = 0.5*(2-8) = -3
    # y1 = 0.5*(3*2 + 0*4) = 3
    W_q = torch.tensor([[1, -2], [3, 0]], dtype=torch.int8, device="cuda")
    x = torch.tensor([2.0, 4.0], dtype=torch.float16, device="cuda")
    y = int8_gemv_fused(W_q, x, 0.5)
    torch.cuda.synchronize()
    assert y[0].item() == pytest.approx(-3.0, abs=1e-3)
    assert y[1].item() == pytest.approx(3.0, abs=1e-3)


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
def test_int8_rejects_non_tensor_scale_shape():
    W_q = torch.randint(-10, 10, (4, 8), dtype=torch.int8, device="cuda")
    x = torch.randn(8, dtype=torch.float16, device="cuda")
    bad_scale = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="numel"):
        int8_gemv_fused(W_q, x, bad_scale)


@requires_cuda
def test_unfused_from_qw_matches_reference():
    torch.manual_seed(3)
    W = torch.randn(16, 64)
    x = torch.randn(64)
    qw = quantize_int8(W, granularity="tensor")
    y_ref = gemv_from_dequant(qw, x)
    y_u = int8_gemv_unfused_from_qw(qw, x.half())
    torch.cuda.synchronize()
    # Unfused CUDA uses FP16 W_hat path; close to Phase-1 reference.
    assert max_abs_error(y_u.cpu(), y_ref) <= _tol(64)


@requires_cuda
def test_fused_closer_traffic_than_materializing_fp16_weights():
    """Sanity: fused API does not require allocating an [N,K] FP16 weight copy."""
    # This is an API/contract check, not a profiler assertion.
    n, k = 32, 128
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw = quantize_int8(W, granularity="tensor")
    y = int8_gemv_fused(qw.qweight.cuda(), x, float(qw.scales.item()))
    assert y.shape == (n,)
    # Reference still available via dequant path for comparison.
    y_ref = fp16_gemv_ref(qw.qweight.float() * float(qw.scales.item()), x.cpu())
    assert max_abs_error(y.cpu(), y_ref) <= _tol(k)
