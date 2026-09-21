"""Tests for error metrics."""

from __future__ import annotations

import pytest
import torch

from qkern.metrics import error_summary, max_abs_error, mean_abs_error, relative_error, rmse


def test_error_metrics_identical():
    a = torch.tensor([1.0, -2.0, 3.5])
    b = a.clone()
    assert max_abs_error(a, b) == 0.0
    assert mean_abs_error(a, b) == 0.0
    assert relative_error(a, b) == 0.0
    assert rmse(a, b) == 0.0


def test_error_metrics_known_values():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 5.0])
    assert max_abs_error(a, b) == pytest.approx(2.0)
    assert mean_abs_error(a, b) == pytest.approx(2.0 / 3.0)
    assert rmse(a, b) == pytest.approx((4.0 / 3.0) ** 0.5)
    # relative: mean(|a-b|/|b|) = (0 + 0 + 2/5) / 3
    assert relative_error(a, b) == pytest.approx((2.0 / 5.0) / 3.0)
    summary = error_summary(a, b)
    assert set(summary) == {
        "max_abs_error",
        "mean_abs_error",
        "relative_error",
        "rmse",
    }
