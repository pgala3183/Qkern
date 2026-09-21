"""
Baseline latency comparison for decode-like FP16 GEMV.

Compares:
  1. PyTorch matmul on CUDA (typically cuBLAS-backed for these shapes)
  2. Explicit cuBLAS-style path via torch.addmm / matmul (same backend family)
  3. QKern naive custom CUDA GEMV

Timing uses torch.cuda.Event (CUDA events). Reports median latency.
Does not claim superiority.
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


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return float(ys[idx])


def _time_cuda(fn, *, warmup: int, iters: int) -> list[float]:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        samples_ms.append(float(starter.elapsed_time(ender)))
    return samples_ms


def _make_inputs(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    W = torch.randn(n, k, dtype=torch.float16, device=device).contiguous()
    x = torch.randn(k, dtype=torch.float16, device=device).contiguous()
    return W, x


def bench_one(n: int, k: int, *, warmup: int, iters: int) -> list[dict]:
    device = torch.device("cuda:0")
    W, x = _make_inputs(n, k, device)
    # Column vector view for matmul baselines: (K,1)
    x_col = x.unsqueeze(1)

    def pytorch_matmul():
        # FP16 inputs; PyTorch/cuBLAS may accumulate in FP16 or TF32-ish paths
        # depending on settings. We keep defaults and report dtype honestly.
        return (W @ x_col).squeeze(1)

    def pytorch_cublas_gemv_like():
        # Same storage as GEMV via explicit matmul; cuBLAS-backed on CUDA.
        return torch.matmul(W, x)

    def qkern_naive():
        return fp16_gemv(W, x)

    rows = []
    for name, fn in [
        ("pytorch_matmul", pytorch_matmul),
        ("pytorch_cublas_backed_matmul", pytorch_cublas_gemv_like),
        ("qkern_fp16_gemv_naive", qkern_naive),
    ]:
        # Warm correctness touch
        out = fn()
        torch.cuda.synchronize()
        assert out.shape == (n,)

        samples = _time_cuda(fn, warmup=warmup, iters=iters)
        rows.append(
            {
                "kernel": name,
                "M": 1,
                "N": n,
                "K": k,
                "dtype": "float16_in_float32_out"
                if name == "qkern_fp16_gemv_naive"
                else "float16",
                "quantization": "none",
                "group_size": None,
                "kernel_config": "naive_one_thread_per_row"
                if name == "qkern_fp16_gemv_naive"
                else "pytorch_default",
                "warmup": warmup,
                "iters": iters,
                "median_latency_ms": _median(samples),
                "min_latency_ms": min(samples),
                "p25_latency_ms": _percentile(samples, 25),
                "p75_latency_ms": _percentile(samples, 75),
                "output_dtype": str(out.dtype).replace("torch.", ""),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="QKern Phase 2 baseline GEMV benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--shapes",
        type=str,
        default="1x1,64x256,256x4096,1024x4096,4096x4096",
        help="Comma-separated NxK shapes",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmarks/results"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; cannot run GPU benchmarks.", file=sys.stderr)
        return 1
    if not is_cuda_extension_available():
        print("qkern CUDA extension is not available; build with pip install -e .", file=sys.stderr)
        return 1

    shapes: list[tuple[int, int]] = []
    for part in args.shapes.split(","):
        n_s, k_s = part.lower().split("x")
        shapes.append((int(n_s), int(k_s)))

    diag = collect_diagnostics()
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qkern_version": __version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "cuda_torch": torch.version.cuda,
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "timing": "torch.cuda.Event elapsed_time median",
        "diagnostics": diag,
        "note": (
            "pytorch_* paths are cuBLAS-backed PyTorch matmul/gemv-like calls. "
            "qkern_fp16_gemv_naive is the handwritten Phase 2 baseline. "
            "No superiority claim is made."
        ),
    }

    all_rows: list[dict] = []
    for n, k in shapes:
        print(f"Benchmarking N={n} K={k} ...")
        all_rows.extend(bench_one(n, k, warmup=args.warmup, iters=args.iters))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.out_dir / f"baseline_fp16_gemv_{stamp}.json"
    csv_path = args.out_dir / f"baseline_fp16_gemv_{stamp}.csv"

    payload = {"meta": meta, "results": all_rows}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
