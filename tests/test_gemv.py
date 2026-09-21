"""Tests for reference FP16 GEMV and quantized unfused GEMV paths."""

from __future__ import annotations

import pytest
import torch

from qkern.metrics import error_summary, max_abs_error
from qkern.quantization import dequantize, quantize_int4, quantize_int8
from qkern.reference import fp16_gemv, gemv_from_dequant, int4_gemv_reference, int8_gemv_reference


@pytest.mark.parametrize(
    "n,k",
    [
        (1, 1),
        (4, 7),
        (8, 32),
        (16, 65),
        (32, 128),
        (3, 257),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_fp16_gemv_matches_fp32_matmul(n, k, seed):
    torch.manual_seed(seed)
    W = torch.randn(n, k, dtype=torch.float16)
    x = torch.randn(k, dtype=torch.float16)
    y = fp16_gemv(W, x)
    # Reference: FP32 matmul of FP16-cast inputs (same cast policy).
    y_ref = (W.float() @ x.float())
    # With FP32 accumulation of FP16 products, should match closely.
    assert y.dtype == torch.float32
    assert y.shape == (n,)
    assert max_abs_error(y, y_ref) < 1e-2 * (1 + k**0.5 * 1e-3) or max_abs_error(y, y_ref) < 1e-2


def test_fp16_gemv_known_small_example():
    W = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]], dtype=torch.float16)
    x = torch.tensor([2.0, 1.0, -1.0], dtype=torch.float16)
    y = fp16_gemv(W, x)
    # y0 = 1*2 + 2*1 + 3*(-1) = 1
    # y1 = -1*2 + 0.5*1 + 0 = -1.5
    assert y[0].item() == pytest.approx(1.0, abs=1e-3)
    assert y[1].item() == pytest.approx(-1.5, abs=1e-3)


def test_fp16_gemv_shape_checks():
    with pytest.raises(ValueError):
        fp16_gemv(torch.randn(2, 3), torch.randn(2, 3))
    with pytest.raises(ValueError):
        fp16_gemv(torch.randn(2, 3), torch.randn(4))


@pytest.mark.parametrize("granularity,group_size", [
    ("tensor", None),
    ("channel", None),
    ("group", 32),
    ("group", 128),
])
@pytest.mark.parametrize("n,k", [(4, 64), (8, 100), (1, 255)])
def test_int8_gemv_reference_close_to_fp(n, k, granularity, group_size):
    torch.manual_seed(3)
    W = torch.randn(n, k) * 0.25
    x = torch.randn(k)
    y_fp = fp16_gemv(W.half(), x.half())
    y_q, qw = int8_gemv_reference(W, x, granularity=granularity, group_size=group_size)
    assert qw.bits == 8
    summary = error_summary(y_q, y_fp)
    # Quantization introduces error; ensure finite and not wildly off.
    assert summary["max_abs_error"] < 5.0
    assert summary["rmse"] < 2.0


@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
@pytest.mark.parametrize("n,k", [(4, 128), (2, 200), (8, 63)])
def test_int4_gemv_packed_reference(n, k, group_size):
    torch.manual_seed(5)
    W = torch.randn(n, k) * 0.25
    x = torch.randn(k)
    y_fp = fp16_gemv(W.half(), x.half())
    y_q, qw = int4_gemv_reference(
        W, x, granularity="group", group_size=group_size, pack=True
    )
    assert qw.bits == 4
    assert qw.packed is True
    assert qw.qweight.dtype == torch.uint8
    # Unfused path via explicit dequant should match helper.
    y2 = gemv_from_dequant(qw, x)
    assert max_abs_error(y_q, y2) == 0.0
    summary = error_summary(y_q, y_fp)
    assert summary["max_abs_error"] < 8.0


def test_int4_primary_target_group128():
    """Primary project target: INT4 weight-only, group 128."""
    torch.manual_seed(11)
    n, k = 16, 256
    W = torch.randn(n, k)
    x = torch.randn(k)
    qw = quantize_int4(W, granularity="group", group_size=128, pack=True)
    assert qw.scales.shape == (n, 2)
    W_hat = dequantize(qw, dtype=torch.float16)
    y = fp16_gemv(W_hat, x.half())
    assert y.shape == (n,)


def test_quantized_gemv_matches_dequant_matmul():
    torch.manual_seed(9)
    W = torch.randn(5, 40)
    x = torch.randn(40)
    qw = quantize_int8(W, granularity="channel")
    y = gemv_from_dequant(qw, x)
    y_ref = dequantize(qw, dtype=torch.float16).float() @ x.float()
    # gemv uses FP16 products + FP32 accum; close to FP32 matmul of dequant weights.
    assert max_abs_error(y, y_ref) < 1e-2
