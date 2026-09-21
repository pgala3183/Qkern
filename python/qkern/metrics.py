"""Error metrics for QKern correctness checks."""

from __future__ import annotations

import torch


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute difference ``max(|a - b|)``."""
    return float((a.float() - b.float()).abs().max().item())


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute difference ``mean(|a - b|)``."""
    return float((a.float() - b.float()).abs().mean().item())


def relative_error(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """
    Mean elementwise relative error.

    ``mean(|a - b| / max(|b|, eps))`` using ``b`` as the reference.
    """
    a_f = a.float()
    b_f = b.float()
    denom = b_f.abs().clamp_min(eps)
    return float(((a_f - b_f).abs() / denom).mean().item())


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Root mean square error ``sqrt(mean((a - b)^2))``."""
    diff = a.float() - b.float()
    return float(torch.sqrt((diff * diff).mean()).item())


def error_summary(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-8) -> dict[str, float]:
    """Return the standard QKern error metric suite."""
    return {
        "max_abs_error": max_abs_error(a, b),
        "mean_abs_error": mean_abs_error(a, b),
        "relative_error": relative_error(a, b, eps=eps),
        "rmse": rmse(a, b),
    }
