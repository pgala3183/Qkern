"""Tests for INT8 / INT4 quantization, dequantization, and packing."""

from __future__ import annotations

import itertools

import pytest
import torch

from qkern.metrics import max_abs_error
from qkern.quantization import (
    INT4_QMAX,
    INT4_QMIN,
    INT8_QMAX,
    INT8_QMIN,
    dequantize,
    dequantize_int4,
    dequantize_int8,
    expand_scales,
    int4_representable_values,
    pack_int4,
    quantize_int4,
    quantize_int8,
    unpack_int4,
)

GROUP_SIZES = [32, 64, 128, 256]


def test_int4_representable_values_complete():
    vals = int4_representable_values()
    assert vals == list(range(-8, 8))
    assert vals[0] == INT4_QMIN
    assert vals[-1] == INT4_QMAX


@pytest.mark.parametrize("a,b", list(itertools.product(range(-8, 8), repeat=2)))
def test_pack_unpack_all_int4_pairs(a: int, b: int):
    """Exhaustively verify all 16x16 = 256 signed INT4 pairs."""
    q = torch.tensor([[a, b]], dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.shape == (1, 1)
    assert packed.dtype == torch.uint8
    # Manual expected packing: low = a, high = b as nibbles.
    expected_byte = (a & 0xF) | ((b & 0xF) << 4)
    assert int(packed[0, 0].item()) == expected_byte
    out = unpack_int4(packed, k=2)
    assert torch.equal(out, q)


def test_pack_unpack_odd_k_padding():
    q = torch.tensor([[-8, 0, 7]], dtype=torch.int8)  # K=3 odd
    packed = pack_int4(q)
    assert packed.shape == (1, 2)
    out = unpack_int4(packed, k=3)
    assert torch.equal(out, q)


def test_pack_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([[8]], dtype=torch.int8))
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([[-9]], dtype=torch.int8))


@pytest.mark.parametrize("bits_fn", [quantize_int8, quantize_int4])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (4, 7),  # odd K
        (8, 32),
        (16, 65),
        (3, 128),
        (5, 200),
    ],
)
@pytest.mark.parametrize("granularity,group_size", [
    ("tensor", None),
    ("channel", None),
    ("group", 32),
    ("group", 64),
    ("group", 128),
    ("group", 256),
])
def test_quant_dequant_shapes_and_roundtrip_bound(bits_fn, shape, granularity, group_size):
    torch.manual_seed(0)
    n, k = shape
    if granularity == "group" and group_size is not None and k < 1:
        pytest.skip("empty")
    W = torch.randn(n, k, dtype=torch.float32) * 0.5
    if bits_fn is quantize_int4:
        qw = bits_fn(W, granularity=granularity, group_size=group_size, pack=False)
        W_hat = dequantize_int4(qw)
        qmax = 7.0
    else:
        qw = bits_fn(W, granularity=granularity, group_size=group_size)
        W_hat = dequantize_int8(qw)
        qmax = 127.0

    assert qw.zero_point == 0
    assert qw.shape == (n, k)
    assert qw.qweight.shape == (n, k)

    # Per-element reconstruction error is at most ~0.5 ULP in quantized domain
    # times scale: |W - W_hat| <= 0.5 * scale_i (+ tiny float noise).
    scales_nk = expand_scales(
        qw.scales, n=n, k=k, granularity=granularity, group_size=group_size
    )
    err = (W - W_hat).abs()
    # Allow slightly over 0.5 due to float32 rounding of W/scale.
    assert torch.all(err <= scales_nk * 0.51 + 1e-6)

    # Scales layout checks.
    if granularity == "tensor":
        assert qw.scales.ndim == 0 or qw.scales.numel() == 1
    elif granularity == "channel":
        assert qw.scales.shape == (n, 1)
    else:
        num_groups = (k + group_size - 1) // group_size
        assert qw.scales.shape == (n, num_groups)

    # Sanity: qmax used in scale definition.
    _ = qmax


@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_group_quant_k_not_divisible(group_size: int):
    torch.manual_seed(1)
    n, k = 4, group_size + 17  # deliberately not divisible
    W = torch.randn(n, k)
    for fn, deq, pack in [
        (quantize_int8, dequantize_int8, False),
        (quantize_int4, dequantize_int4, True),
    ]:
        kwargs = {"granularity": "group", "group_size": group_size}
        if fn is quantize_int4:
            kwargs["pack"] = pack
        qw = fn(W, **kwargs)
        W_hat = deq(qw)
        assert W_hat.shape == (n, k)
        assert max_abs_error(W_hat, dequantize(qw)) == 0.0


def test_int8_range_clipping():
    W = torch.tensor([[1000.0, -1000.0]])
    qw = quantize_int8(W, granularity="tensor")
    assert int(qw.qweight.min()) >= INT8_QMIN
    assert int(qw.qweight.max()) <= INT8_QMAX
    # With scale = 1000/127, +/-1000 maps near +/-127.
    assert int(qw.qweight[0, 0]) == 127
    assert int(qw.qweight[0, 1]) == -127


def test_int4_range_clipping():
    W = torch.tensor([[100.0, -100.0]])
    qw = quantize_int4(W, granularity="tensor", pack=False)
    assert int(qw.qweight.min()) >= INT4_QMIN
    assert int(qw.qweight.max()) <= INT4_QMAX
    assert int(qw.qweight[0, 0]) == 7
    assert int(qw.qweight[0, 1]) == -7


def test_zero_weights():
    W = torch.zeros(3, 5)
    qw8 = quantize_int8(W, granularity="channel")
    assert torch.all(qw8.qweight == 0)
    assert torch.allclose(dequantize_int8(qw8), W)
    qw4 = quantize_int4(W, granularity="group", group_size=2, pack=True)
    assert torch.allclose(dequantize_int4(qw4), W)


def test_per_channel_scales_differ():
    W = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [10.0, -10.0, 5.0],
        ],
        dtype=torch.float32,
    )
    qw = quantize_int8(W, granularity="channel")
    assert qw.scales.shape == (2, 1)
    assert qw.scales[1, 0] > qw.scales[0, 0]


def test_per_group_scales_layout():
    W = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    qw = quantize_int8(W, granularity="group", group_size=4)
    assert qw.scales.shape == (1, 4)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
@pytest.mark.parametrize("n,k", [(2, 9), (7, 63), (1, 257)])
def test_random_int4_packed_roundtrip_storage(seed, n, k):
    torch.manual_seed(seed)
    # Build a random INT4 tensor directly, pack/unpack must be lossless.
    q = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.dtype == torch.uint8
    assert packed.shape == (n, (k + 1) // 2)
    assert torch.equal(unpack_int4(packed, k=k), q)


def test_invalid_api_args():
    W = torch.randn(2, 8)
    with pytest.raises(ValueError):
        quantize_int8(W, granularity="group", group_size=None)
    with pytest.raises(ValueError):
        quantize_int8(W, granularity="tensor", group_size=32)
    with pytest.raises(ValueError):
        quantize_int8(torch.randn(2, 3, 4), granularity="tensor")
