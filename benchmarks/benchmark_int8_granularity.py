"""
INT8 fused GEMV granularity benchmark.

Table columns: granularity | group size | latency | error | logical bandwidth

Does not pick a "best" granularity — reports measured tradeoffs only.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from qkern import (
    __version__,
    gemv_from_dequant,
    int8_gemv_fused_from_qw,
    is_cuda_extension_available,
    max_abs_error,
    quantize_int8,
    rmse,
)
from qkern.diagnostics import collect_diagnostics

GROUP_SIZES = [32, 64, 128, 256]


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def time_cuda(fn, *, warmup: int, iters: int) -> list[float]:
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


def logical_bytes(n: int, k: int, granularity: str, group_size: int | None) -> float:
    w = float(n * k)  # INT8 weights
    x = float(k * 2)
    y = float(n * 4)
    if granularity == "tensor":
        scales = 4.0
    elif granularity == "channel":
        scales = float(n * 4)
    else:
        assert group_size is not None
        num_groups = (k + group_size - 1) // group_size
        scales = float(n * num_groups * 4)
    return w + x + y + scales


def configs():
    yield ("tensor", None)
    yield ("channel", None)
    for gs in GROUP_SIZES:
        yield ("group", gs)


def bench_one(n: int, k: int, *, warmup: int, iters: int) -> list[dict]:
    torch.manual_seed(0)
    W = torch.randn(n, k)
    x_cpu = torch.randn(k)
    x = x_cpu.half().cuda().contiguous()

    rows: list[dict] = []
    for granularity, group_size in configs():
        qw = quantize_int8(W, granularity=granularity, group_size=group_size)
        y_ref = gemv_from_dequant(qw, x_cpu)

        def fn(q=qw):
            return int8_gemv_fused_from_qw(q, x)

        y = fn()
        torch.cuda.synchronize()
        samples = time_cuda(fn, warmup=warmup, iters=iters)
        med = _median(samples)
        med_s = med * 1e-3
        lb = logical_bytes(n, k, granularity, group_size)
        rows.append(
            {
                "granularity": granularity,
                "group_size": group_size if group_size is not None else "",
                "N": n,
                "K": k,
                "median_latency_ms": med,
                "max_abs_error": max_abs_error(y.cpu(), y_ref),
                "rmse": rmse(y.cpu(), y_ref),
                "logical_bytes_total": lb,
                "logical_bandwidth_GBs": lb / med_s / 1e9 if med_s > 0 else float("nan"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--shapes", default="64x256,256x4096,1024x4096")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    if not torch.cuda.is_available() or not is_cuda_extension_available():
        print("CUDA + qkern extension required", file=sys.stderr)
        return 1

    shapes = []
    for part in args.shapes.split(","):
        n_s, k_s = part.lower().split("x")
        shapes.append((int(n_s), int(k_s)))

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qkern_version": __version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "timing": "torch.cuda.Event median",
        "experiment": "int8_gemv_granularity",
        "note": "No granularity is declared best; table reports measured tradeoffs.",
        "diagnostics": collect_diagnostics(),
    }

    results: list[dict] = []
    for n, k in shapes:
        print(f"Benchmarking N={n} K={k} ...")
        results.extend(bench_one(n, k, warmup=args.warmup, iters=args.iters))

    # Print markdown-friendly table for the primary shape (last or first large).
    print("\ngranularity | group_size | N | K | latency_ms | max_abs_error | logical_bw_GBs")
    print("--- | --- | --- | --- | --- | --- | ---")
    for r in results:
        print(
            f"{r['granularity']} | {r['group_size']} | {r['N']} | {r['K']} | "
            f"{r['median_latency_ms']:.4f} | {r['max_abs_error']:.6g} | {r['logical_bandwidth_GBs']:.3f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.out_dir / f"int8_gemv_granularity_{stamp}.json"
    csv_path = args.out_dir / f"int8_gemv_granularity_{stamp}.csv"
    payload = {"meta": meta, "results": results}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
