"""
Fused vs unfused INT4 (W4A16, group_size=128) experiment.

Unfused: INT4 → CUDA dequant → FP16 Ŵ → FP16 GEMV
Fused:   INT4 + scales + x → single fused GEMV kernel

Measures (CUDA events, identical sync methodology):
  - dequantization latency
  - GEMV latency (on already-materialized Ŵ)
  - total unfused latency (dequant + GEMV in one timed region)
  - fused latency
  - logical bytes
  - speedup = unfused_total / fused

Does not assume fusion always wins.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from qkern import (
    __version__,
    fp16_gemv_vec2,
    gemv_from_dequant,
    int4_dequant_from_qw,
    int4_gemv_fused_from_qw,
    int4_gemv_unfused_from_qw,
    is_cuda_extension_available,
    max_abs_error,
    quantize_int4,
)
from qkern.diagnostics import collect_diagnostics

DEFAULT_SHAPES = [
    (4096, 4096),
    (11008, 4096),
    (4096, 11008),
    (14336, 4096),
]
GROUP_SIZE = 128


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def _percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return float(ys[idx])


def time_cuda(fn, *, warmup: int, iters: int) -> list[float]:
    """Median-friendly CUDA-event timing with synchronize after each sample."""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        samples.append(float(starter.elapsed_time(ender)))
    return samples


def logical_bytes_fused(n: int, k: int, group_size: int) -> dict[str, float]:
    packed = float(n * ((k + 1) // 2))
    num_groups = (k + group_size - 1) // group_size
    scales = float(n * num_groups * 4)
    x = float(k * 2)
    y = float(n * 4)
    total = packed + scales + x + y
    return {
        "packed_weight_bytes": packed,
        "scale_bytes": scales,
        "activation_bytes": x,
        "output_bytes": y,
        "dequant_write_bytes": 0.0,
        "dequant_read_as_fp16_bytes": 0.0,
        "total_logical_bytes": total,
    }


def logical_bytes_unfused(n: int, k: int, group_size: int) -> dict[str, float]:
    packed = float(n * ((k + 1) // 2))
    num_groups = (k + group_size - 1) // group_size
    scales = float(n * num_groups * 4)
    w_hat = float(n * k * 2)  # FP16 materialization
    x = float(k * 2)
    y = float(n * 4)
    # Dequant reads packed+scales, writes Ŵ; GEMV reads Ŵ + x, writes y.
    total = packed + scales + w_hat + w_hat + x + y
    return {
        "packed_weight_bytes": packed,
        "scale_bytes": scales,
        "activation_bytes": x,
        "output_bytes": y,
        "dequant_write_bytes": w_hat,
        "dequant_read_as_fp16_bytes": w_hat,
        "total_logical_bytes": total,
    }


def bench_shape(
    n: int,
    k: int,
    *,
    group_size: int,
    warmup: int,
    iters: int,
) -> dict:
    torch.manual_seed(0)
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda").contiguous()
    qw = quantize_int4(W, granularity="group", group_size=group_size, pack=True)
    qw_cuda = replace(
        qw,
        qweight=qw.qweight.cuda().contiguous(),
        scales=qw.scales.to(device="cuda", dtype=torch.float32).contiguous(),
    )

    y_ref = gemv_from_dequant(qw, x.cpu().float())

    def fused():
        return int4_gemv_fused_from_qw(qw_cuda, x)

    def dequant_only():
        return int4_dequant_from_qw(qw_cuda)

    # Materialize once for GEMV-only timing (same Ŵ both warmups/iters reuse).
    W_hat = int4_dequant_from_qw(qw_cuda)
    torch.cuda.synchronize()

    def gemv_only():
        return fp16_gemv_vec2(W_hat, x)

    def unfused_total():
        return int4_gemv_unfused_from_qw(qw_cuda, x)

    # Warm both end-to-end paths before measured runs (shared methodology).
    for _ in range(warmup):
        fused()
        unfused_total()
    torch.cuda.synchronize()

    # Stage timings: each call uses the same event+sync pattern (warmup=0 here
    # because we already warmed both full paths above; still warm each stage
    # lightly for cache fairness).
    stage_warmup = max(5, warmup // 4)
    dequant_samples = time_cuda(dequant_only, warmup=stage_warmup, iters=iters)
    gemv_samples = time_cuda(gemv_only, warmup=stage_warmup, iters=iters)
    unfused_samples = time_cuda(unfused_total, warmup=stage_warmup, iters=iters)
    fused_samples = time_cuda(fused, warmup=stage_warmup, iters=iters)

    y_f = fused()
    y_u = unfused_total()
    torch.cuda.synchronize()
    err_f = float(max_abs_error(y_f.cpu(), y_ref))
    err_u = float(max_abs_error(y_u.cpu(), y_ref))

    deq_ms = _median(dequant_samples)
    gemv_ms = _median(gemv_samples)
    unfused_ms = _median(unfused_samples)
    fused_ms = _median(fused_samples)
    speedup = (unfused_ms / fused_ms) if fused_ms > 0 else float("nan")

    lb_f = logical_bytes_fused(n, k, group_size)
    lb_u = logical_bytes_unfused(n, k, group_size)

    return {
        "M": 1,
        "N": n,
        "K": k,
        "group_size": group_size,
        "quantization": "int4_group",
        "dequant_latency_median_ms": deq_ms,
        "dequant_latency_min_ms": float(min(dequant_samples)),
        "dequant_latency_p25_ms": _percentile(dequant_samples, 25),
        "dequant_latency_p75_ms": _percentile(dequant_samples, 75),
        "gemv_latency_median_ms": gemv_ms,
        "gemv_latency_min_ms": float(min(gemv_samples)),
        "unfused_total_latency_median_ms": unfused_ms,
        "unfused_total_latency_min_ms": float(min(unfused_samples)),
        "unfused_sum_stages_ms": deq_ms + gemv_ms,
        "fused_latency_median_ms": fused_ms,
        "fused_latency_min_ms": float(min(fused_samples)),
        "speedup_unfused_over_fused": speedup,
        "fused_max_abs_error": err_f,
        "unfused_max_abs_error": err_u,
        "logical_bytes_fused": lb_f["total_logical_bytes"],
        "logical_bytes_unfused": lb_u["total_logical_bytes"],
        "logical_bytes_extra_unfused": lb_u["total_logical_bytes"] - lb_f["total_logical_bytes"],
        "logical_detail_fused": lb_f,
        "logical_detail_unfused": lb_u,
        "warmup": warmup,
        "iters": iters,
        "timing": "torch.cuda.Event + synchronize per sample",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--group-size", type=int, default=GROUP_SIZE)
    p.add_argument(
        "--shapes",
        default="4096x4096,11008x4096,4096x11008,14336x4096",
        help="Comma-separated NxK",
    )
    args = p.parse_args()

    if not torch.cuda.is_available() or not is_cuda_extension_available():
        print("CUDA + qkern._C required", file=sys.stderr)
        return 1

    shapes: list[tuple[int, int]] = []
    for part in args.shapes.split(","):
        n_s, k_s = part.lower().split("x")
        shapes.append((int(n_s), int(k_s)))

    gpu = torch.cuda.get_device_name(0)
    cc = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    print(f"qkern {__version__} | fused vs unfused INT4 group={args.group_size}")
    print(f"GPU: {gpu} (cc {cc})")
    print(
        f"{'N':>6} {'K':>6} {'deq_ms':>8} {'gemv_ms':>8} {'unf_ms':>8} "
        f"{'fus_ms':>8} {'speedup':>8} {'extra_B':>12}"
    )

    rows: list[dict] = []
    for n, k in shapes:
        print(f"Benchmarking N={n} K={k} ...", flush=True)
        row = bench_shape(
            n, k, group_size=args.group_size, warmup=args.warmup, iters=args.iters
        )
        row.update(
            {
                "gpu": gpu,
                "compute_capability": cc,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "qkern_version": __version__,
            }
        )
        print(
            f"{row['N']:6d} {row['K']:6d} "
            f"{row['dequant_latency_median_ms']:8.3f} "
            f"{row['gemv_latency_median_ms']:8.3f} "
            f"{row['unfused_total_latency_median_ms']:8.3f} "
            f"{row['fused_latency_median_ms']:8.3f} "
            f"{row['speedup_unfused_over_fused']:8.3f} "
            f"{row['logical_bytes_extra_unfused']:12.0f}"
        )
        rows.append(row)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"fused_vs_unfused_int4_{stamp}.json"
    csv_path = out_dir / f"fused_vs_unfused_int4_{stamp}.csv"

    # Flatten nested dicts for CSV.
    csv_rows = []
    for r in rows:
        flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for prefix, detail in (
            ("fused", r["logical_detail_fused"]),
            ("unfused", r["logical_detail_unfused"]),
        ):
            for dk, dv in detail.items():
                flat[f"{prefix}_{dk}"] = dv
        csv_rows.append(flat)

    payload = {
        "meta": {
            "experiment": "fused_vs_unfused_int4_group128",
            "M": 1,
            "group_size": args.group_size,
            "gpu": gpu,
            "compute_capability": cc,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "qkern_version": __version__,
            "warmup": args.warmup,
            "iters": args.iters,
            "timing": "torch.cuda.Event + synchronize per sample",
            "paths": {
                "unfused": "int4_dequant (CUDA) -> fp16_gemv_vec2",
                "fused": "int4_gemv_fused (group)",
            },
            "note": "Speedup = unfused_total / fused. Fusion is not assumed to win.",
            "diagnostics": collect_diagnostics(),
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
