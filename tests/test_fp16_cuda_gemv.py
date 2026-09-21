"""Correctness tests for CUDA FP16 GEMV variants vs Phase 1 reference."""

from __future__ import annotations

import pytest
import torch

from qkern import (
    fp16_gemv,
    fp16_gemv_naive,
    fp16_gemv_ref,
    fp16_gemv_vec2,
    fp16_gemv_x_smem,
    is_cuda_extension_available,
)
from qkern.metrics import error_summary, max_abs_error

requires_cuda = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not is_cuda_extension_available()),
    reason="CUDA device + qkern._C extension required",
)

VARIANTS = ["naive", "x_smem", "vec2"]


def _tol_for_shape(k: int) -> float:
    return max(5e-2, 2e-3 * (k ** 0.5))


@requires_cuda
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "n,k",
    [
        (1, 1),  # small K
        (1, 2),
        (4, 7),  # odd / non-aligned-friendly K
        (8, 32),  # aligned K
        (16, 65),  # odd large-ish
        (64, 256),
        (128, 1024),
        (256, 4096),  # large K
        (1024, 4096),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_cuda_fp16_gemv_matches_reference(variant, n, k, seed):
    torch.manual_seed(seed)
    W = torch.randn(n, k, dtype=torch.float16)
    x = torch.randn(k, dtype=torch.float16)

    y_ref = fp16_gemv_ref(W, x)
    y_cuda = fp16_gemv(W.cuda().contiguous(), x.cuda().contiguous(), variant=variant)
    torch.cuda.synchronize()

    assert y_cuda.dtype == torch.float32
    assert y_cuda.shape == (n,)
    err = max_abs_error(y_cuda.cpu(), y_ref)
    assert err <= _tol_for_shape(k), (variant, error_summary(y_cuda.cpu(), y_ref))


@requires_cuda
@pytest.mark.parametrize("k", [1, 2, 3, 7, 8, 31, 32, 63, 64, 128, 257, 4096])
def test_vec2_aligned_and_nonaligned_k(k):
    """Aligned (even K) and non-aligned (odd K) coverage for vec2."""
    torch.manual_seed(k)
    n = 16
    W_cpu = torch.randn(n, k, dtype=torch.float16)
    x_cpu = torch.randn(k, dtype=torch.float16)
    W = W_cpu.cuda().contiguous()
    x = x_cpu.cuda().contiguous()
    y_v = fp16_gemv_vec2(W, x)
    y_ref = fp16_gemv_ref(W_cpu, x_cpu)
    torch.cuda.synchronize()
    # half2 convert path can differ slightly from pure scalar at large K.
    assert max_abs_error(y_v.cpu(), y_ref) <= _tol_for_shape(k)


@requires_cuda
def test_vec2_misaligned_x_view_fallback():
    """Force x to start at a 2-mod-4 address via a storage offset view."""
    torch.manual_seed(0)
    n, k = 32, 128
    W = torch.randn(n, k, dtype=torch.float16, device="cuda").contiguous()
    buf = torch.randn(k + 1, dtype=torch.float16, device="cuda")
    x = buf[1:].contiguous()  # still contiguous; base may be 2B-aligned only
    assert x.shape == (k,)
    y_v = fp16_gemv_vec2(W, x)
    y_n = fp16_gemv_naive(W, x)
    torch.cuda.synchronize()
    assert max_abs_error(y_v, y_n) < 1e-4


@requires_cuda
def test_variants_agree_on_small_example():
    W = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]], dtype=torch.float16, device="cuda")
    x = torch.tensor([2.0, 1.0, -1.0], dtype=torch.float16, device="cuda")
    y_n = fp16_gemv_naive(W, x)
    y_s = fp16_gemv_x_smem(W, x)
    y_v = fp16_gemv_vec2(W, x)
    torch.cuda.synchronize()
    assert max_abs_error(y_n, y_s) < 1e-5
    assert max_abs_error(y_n, y_v) < 1e-5
    assert y_n[0].item() == pytest.approx(1.0, abs=1e-3)
    assert y_n[1].item() == pytest.approx(-1.5, abs=1e-3)


@requires_cuda
def test_cuda_fp16_gemv_rejects_cpu():
    W = torch.randn(4, 8, dtype=torch.float16)
    x = torch.randn(8, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="CUDA"):
        fp16_gemv(W, x)


@requires_cuda
def test_cuda_fp16_gemv_rejects_wrong_dtype():
    W = torch.randn(4, 8, dtype=torch.float32, device="cuda")
    x = torch.randn(8, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="float16"):
        fp16_gemv(W, x)


@requires_cuda
def test_cuda_fp16_gemv_rejects_noncontiguous():
    base = torch.randn(4, 16, dtype=torch.float16, device="cuda")
    W = base[:, ::2]
    assert not W.is_contiguous()
    x = torch.randn(8, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="contiguous"):
        fp16_gemv(W, x)


@requires_cuda
def test_cuda_fp16_gemv_rejects_shape_mismatch():
    W = torch.randn(4, 8, dtype=torch.float16, device="cuda")
    x = torch.randn(7, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="K mismatch"):
        fp16_gemv(W, x)


@requires_cuda
def test_unknown_variant_rejected():
    W = torch.randn(4, 8, dtype=torch.float16, device="cuda")
    x = torch.randn(8, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="unknown variant"):
        fp16_gemv(W, x, variant="not_a_real_variant")  # type: ignore[arg-type]
