"""
INT8 fused vs unfused GEMV benchmark (per-tensor quantization).

Compares:
  1. PyTorch / cuBLAS-backed FP16 matmul
  2. Custom FP16 GEMV (vec2)
  3. Unfused INT8 (dequant to FP16 tensor + custom FP16 GEMV)
  4. Fused INT8 (custom kernel; no dequantized W in global memory)

Does not claim INT8 must be faster — measurements decide.
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
    fp16_gemv_vec2,
    int8_gemv_fused,
    int8_gemv_unfused,
    is_cuda_extension_available,
    quantize_int8,
)
from qkern.diagnostics import collect_diagnostics


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def _percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return float(ys[idx])


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


def logical_weight_bytes(n: int, k: int, kind: str) -> float:
    if kind in ("pytorch_fp16", "qkern_fp16_vec2"):
        return float(n * k * 2)  # FP16 weights
    if kind == "int8_unfused":
        # INT8 source + materialized FP16 dequant buffer touched in the path.
        return float(n * k * 1 + n * k * 2)
    if kind == "int8_fused":
        return float(n * k * 1)  # INT8 weights only (no dequant matrix)
    raise ValueError(kind)


def bench_shape(n: int, k: int, *, warmup: int, iters: int) -> list[dict]:
    torch.manual_seed(0)
    W_fp = torch.randn(n, k, dtype=torch.float16, device="cuda").contiguous()
    x = torch.randn(k, dtype=torch.float16, device="cuda").contiguous()
    qw = quantize_int8(W_fp.float().cpu(), granularity="tensor")
    W_q = qw.qweight.to(device="cuda").contiguous()
    scale = float(qw.scales.item())

    def pytorch_fp16():
        return torch.matmul(W_fp, x)

    def qkern_fp16():
        return fp16_gemv_vec2(W_fp, x)

    def int8_unfused():
        return int8_gemv_unfused(W_q, x, scale, fp16_variant="vec2")

    def int8_fused():
        return int8_gemv_fused(W_q, x, scale)

    rows: list[dict] = []
    fp16_med = None
    for name, fn in [
        ("pytorch_fp16", pytorch_fp16),
        ("qkern_fp16_vec2", qkern_fp16),
        ("int8_unfused", int8_unfused),
        ("int8_fused", int8_fused),
    ]:
        out = fn()
        torch.cuda.synchronize()
        assert out.shape == (n,)
        samples = time_cuda(fn, warmup=warmup, iters=iters)
        med = _median(samples)
        if name == "pytorch_fp16":
            fp16_med = med
        speedup = (fp16_med / med) if (fp16_med and med > 0) else float("nan")
        w_bytes = logical_weight_bytes(n, k, name)
        # Activations + output (same for all): K*2 + N*4
        total_logical = w_bytes + float(k * 2 + n * 4)
        med_s = med * 1e-3
        logical_bw = total_logical / med_s / 1e9 if med_s > 0 else float("nan")
        rows.append(
            {
                "kernel": name,
                "M": 1,
                "N": n,
                "K": k,
                "dtype": "int8_w_fp16_x" if "int8" in name else "float16",
                "quantization": "int8_per_tensor" if "int8" in name else "none",
                "group_size": None,
                "warmup": warmup,
                "iters": iters,
                "median_latency_ms": med,
                "min_latency_ms": min(samples),
                "p25_latency_ms": _percentile(samples, 25),
                "p75_latency_ms": _percentile(samples, 75),
                "speedup_vs_pytorch_fp16": speedup,
                "logical_weight_bytes": w_bytes,
                "logical_bytes_total": total_logical,
                "logical_bandwidth_GBs": logical_bw,
                "output_dtype": str(out.dtype).replace("torch.", ""),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--shapes", default="64x256,256x4096,1024x4096,4096x4096")
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
        "experiment": "int8_gemv_fused_per_tensor",
        "note": (
            "Fusion avoids writing a dequantized [N,K] FP16/FP32 weight matrix. "
            "INT8 is not assumed faster than FP16; report measured speedups only."
        ),
        "diagnostics": collect_diagnostics(),
    }

    results: list[dict] = []
    for n, k in shapes:
        print(f"Benchmarking N={n} K={k} ...")
        results.extend(bench_shape(n, k, warmup=args.warmup, iters=args.iters))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.out_dir / f"int8_gemv_fused_{stamp}.json"
    csv_path = args.out_dir / f"int8_gemv_fused_{stamp}.csv"
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
