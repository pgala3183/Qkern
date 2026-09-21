"""
Compare FP16 GEMV variants: naive, x_smem, vec2.

Identical CUDA-event methodology. Reports median latency, speedup vs naive,
logical bytes, logical bandwidth. Does not claim superiority beyond measurements.
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

from qkern import __version__, fp16_gemv, is_cuda_extension_available
from qkern.diagnostics import collect_diagnostics

VARIANTS = ("naive", "x_smem", "vec2")


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def _percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return float(ys[idx])


def logical_bytes(n: int, k: int, variant: str) -> dict[str, float]:
    w = float(n * k * 2)
    y = float(n * 4)
    if variant in ("naive", "vec2"):
        # Same implied x traffic model as naive (each row-thread reads x).
        x = float(n * k * 2)
    elif variant == "x_smem":
        x = float(k * 2)
    else:
        raise ValueError(variant)
    total = w + x + y
    return {
        "logical_bytes_W": w,
        "logical_bytes_x": x,
        "logical_bytes_y": y,
        "logical_bytes_total": total,
    }


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


def bench_shape(n: int, k: int, *, warmup: int, iters: int) -> list[dict]:
    W = torch.randn(n, k, dtype=torch.float16, device="cuda").contiguous()
    x = torch.randn(k, dtype=torch.float16, device="cuda").contiguous()

    rows: list[dict] = []
    naive_med = None
    for variant in VARIANTS:

        def fn(v=variant):
            return fp16_gemv(W, x, variant=v)

        out = fn()
        torch.cuda.synchronize()
        assert out.shape == (n,)

        samples = time_cuda(fn, warmup=warmup, iters=iters)
        med = _median(samples)
        if variant == "naive":
            naive_med = med
        speedup = (naive_med / med) if (naive_med and med > 0) else float("nan")
        lb = logical_bytes(n, k, variant)
        med_s = med * 1e-3
        logical_bw = lb["logical_bytes_total"] / med_s / 1e9 if med_s > 0 else float("nan")

        rows.append(
            {
                "kernel": f"fp16_gemv_{variant}",
                "variant": variant,
                "M": 1,
                "N": n,
                "K": k,
                "dtype": "float16_in_float32_out",
                "warmup": warmup,
                "iters": iters,
                "median_latency_ms": med,
                "min_latency_ms": min(samples),
                "p25_latency_ms": _percentile(samples, 25),
                "p75_latency_ms": _percentile(samples, 75),
                "speedup_vs_naive": speedup,
                **lb,
                "logical_bandwidth_GBs": logical_bw,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--shapes",
        default="64x256,256x4096,1024x4096,4096x4096,16x65,16x64",
    )
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
        "cuda_torch": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "timing": "torch.cuda.Event median",
        "experiment": "fp16_gemv_vec2",
        "variants": list(VARIANTS),
        "diagnostics": collect_diagnostics(),
    }

    results: list[dict] = []
    for n, k in shapes:
        print(f"Benchmarking N={n} K={k} ...")
        results.extend(bench_shape(n, k, warmup=args.warmup, iters=args.iters))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.out_dir / f"fp16_gemv_vec2_{stamp}.json"
    csv_path = args.out_dir / f"fp16_gemv_vec2_{stamp}.csv"
    payload = {"meta": meta, "results": results}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
