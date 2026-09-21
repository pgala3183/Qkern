"""Correctness tests for compile-time INT4 GEMV configurations."""

from __future__ import annotations

import pytest
import torch

from qkern import (
    gemv_from_dequant,
    int4_gemv_fused_from_qw,
    int4_gemv_list_configs,
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
def test_int4_list_configs_has_default_and_three():
    configs = int4_gemv_list_configs()
    assert len(configs) >= 3
    names = [c["name"] for c in configs]
    assert "default" in names
    assert any(c["is_default"] for c in configs)
    for c in configs:
        assert c["BLOCK_M"] == 1
        assert c["BLOCK_N"] > 0
        assert c["VEC_SIZE"] in (1, 2, 4)
        assert c["NUM_STAGES"] >= 1


@requires_cuda
@pytest.mark.parametrize(
    "config",
    ["default", "bn128_bk128_v1_s1", "bn256_bk256_v4_s2"],
)
@pytest.mark.parametrize(
    "n,k",
    [
        (8, 7),
        (16, 65),
        (64, 256),
        (128, 1024),
        (256, 4096),
    ],
)
def test_int4_configs_match_reference_group128(config, n, k):
    torch.manual_seed(0)
    W = torch.randn(n, k) * 0.5
    x = torch.randn(k)
    qw = quantize_int4(W, granularity="group", group_size=128, pack=True)
    y_ref = gemv_from_dequant(qw, x)
    y = int4_gemv_fused_from_qw(qw, x.half(), config=config)
    torch.cuda.synchronize()
    assert max_abs_error(y.cpu(), y_ref) <= _tol(k)


@requires_cuda
def test_int4_configs_agree_with_each_other():
    torch.manual_seed(1)
    n, k = 96, 512
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw = quantize_int4(W, granularity="group", group_size=128, pack=True)
    ys = [
        int4_gemv_fused_from_qw(qw, x, config=c).cpu()
        for c in ("default", "bn128_bk128_v1_s1", "bn256_bk256_v4_s2")
    ]
    torch.cuda.synchronize()
    assert max_abs_error(ys[0], ys[1]) <= _tol(k)
    assert max_abs_error(ys[0], ys[2]) <= _tol(k)


@requires_cuda
def test_int4_unknown_config_rejected():
    torch.manual_seed(0)
    qw = quantize_int4(torch.randn(4, 32), granularity="tensor", pack=True)
    x = torch.randn(32, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="unknown config"):
        int4_gemv_fused_from_qw(qw, x, config="not_a_real_config")
