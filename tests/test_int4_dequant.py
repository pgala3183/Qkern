"""Correctness tests for CUDA INT4 dequant and fused vs unfused GEMV."""

from __future__ import annotations

import pytest
import torch

from qkern import (
    dequantize,
    gemv_from_dequant,
    int4_dequant_from_qw,
    int4_gemv_fused_from_qw,
    int4_gemv_unfused_from_qw,
    is_cuda_extension_available,
    max_abs_error,
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
@pytest.mark.parametrize("n,k", [(4, 7), (8, 65), (16, 128), (32, 257)])
def test_int4_dequant_vs_pytorch(granularity, group_size, n, k):
    torch.manual_seed(0)
    W = torch.randn(n, k)
    qw = quantize_int4(W, granularity=granularity, group_size=group_size, pack=True)
    ref = dequantize(qw, dtype=torch.float16)
    got = int4_dequant_from_qw(qw)
    torch.cuda.synchronize()
    # FP16 store round-trip: allow small ulp differences vs CPU float→half.
    assert max_abs_error(got.cpu().float(), ref.float()) <= 2e-3


@requires_cuda
def test_fused_vs_unfused_group128():
    torch.manual_seed(1)
    n, k = 64, 4096
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw = quantize_int4(W, granularity="group", group_size=128, pack=True)
    y_f = int4_gemv_fused_from_qw(qw, x)
    y_u = int4_gemv_unfused_from_qw(qw, x)
    y_ref = gemv_from_dequant(qw, x.cpu().float())
    torch.cuda.synchronize()
    assert max_abs_error(y_f.cpu(), y_ref) <= _tol(k)
    assert max_abs_error(y_u.cpu(), y_ref) <= _tol(k)
    assert max_abs_error(y_f.cpu(), y_u.cpu()) <= _tol(k)
