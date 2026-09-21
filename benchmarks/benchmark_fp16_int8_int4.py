"""
Compare FP16 / INT8 / INT4 GEMV paths (per-tensor quantization).

Reports latency, logical weight bytes, logical bandwidth, and max abs error.
Does not declare a winner — measurements only.
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
    gemv_from_dequant,
    int4_gemv_fused,
    int8_gemv_fused,
    is_cuda_extension_available,
    max_abs_error,
    quantize_int4,
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
        return float(n * k * 2)
    if kind == "int8_fused":
        return float(n * k * 1)
    if kind == "int4_fused":
        return float(n * ((k + 1) // 2))  # packed uint8
    raise ValueError(kind)


def bench_shape(n: int, k: int, *, warmup: int, iters: int) -> list[dict]:
    torch.manual_seed(0)
    W_fp = torch.randn(n, k, dtype=torch.float16, device="cuda").contiguous()
    x = torch.randn(k, dtype=torch.float16, device="cuda").contiguous()

    qw8 = quantize_int8(W_fp.float().cpu(), granularity="tensor")
    W8 = qw8.qweight.to(device="cuda").contiguous()
    s8 = float(qw8.scales.item())

    qw4 = quantize_int4(W_fp.float().cpu(), granularity="tensor", pack=True)
    W4 = qw4.qweight.to(device="cuda").contiguous()
    s4 = float(qw4.scales.item())

    y_ref8 = gemv_from_dequant(qw8, x.cpu().float())
    y_ref4 = gemv_from_dequant(qw4, x.cpu().float())
    y_ref_fp = (W_fp.float().cpu() @ x.float().cpu())

    def pytorch_fp16():
        return torch.matmul(W_fp, x)

    def qkern_fp16():
        return fp16_gemv_vec2(W_fp, x)

    def int8_fused():
        return int8_gemv_fused(W8, x, s8)

    def int4_fused():
        return int4_gemv_fused(W4, x, s4, K=k)

    refs = {
        "pytorch_fp16": y_ref_fp,
        "qkern_fp16_vec2": y_ref_fp,
        "int8_fused": y_ref8,
        "int4_fused": y_ref4,
    }

    rows: list[dict] = []
    for name, fn in [
        ("pytorch_fp16", pytorch_fp16),
        ("qkern_fp16_vec2", qkern_fp16),
        ("int8_fused", int8_fused),
        ("int4_fused", int4_fused),
    ]:
        out = fn()
        torch.cuda.synchronize()
        assert out.shape == (n,)
        err = float(max_abs_error(out.detach().float().cpu(), refs[name]))
        samples = time_cuda(fn, warmup=warmup, iters=iters)
        med = _median(samples)
        w_bytes = logical_weight_bytes(n, k, name)
        total_logical = w_bytes + float(k * 2 + n * 4)  # x FP16 + y FP32
        med_s = med * 1e-3
        logical_bw = total_logical / med_s / 1e9 if med_s > 0 else float("nan")
        rows.append(
            {
                "kernel": name,
                "M": 1,
                "N": n,
                "K": k,
                "dtype_weights": {
                    "pytorch_fp16": "fp16",
                    "qkern_fp16_vec2": "fp16",
                    "int8_fused": "int8",
                    "int4_fused": "int4_packed",
                }[name],
                "quantization": "tensor" if "int" in name else "none",
                "group_size": None,
                "warmup": warmup,
                "iters": iters,
                "latency_median_ms": med,
                "latency_min_ms": float(min(samples)),
                "latency_p25_ms": _percentile(samples, 25),
                "latency_p75_ms": _percentile(samples, 75),
                "logical_weight_bytes": w_bytes,
                "logical_total_bytes": total_logical,
                "logical_bandwidth_GBps": logical_bw,
                "max_abs_error": err,
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--shapes",
        nargs="*",
        default=["64x256", "256x4096", "1024x4096"],
        help="NxK shapes",
    )
    args = p.parse_args()

    if not torch.cuda.is_available() or not is_cuda_extension_available():
        print("CUDA + qkern._C required", file=sys.stderr)
        return 1

    diag = collect_diagnostics()
    shapes: list[tuple[int, int]] = []
    for s in args.shapes:
        n_s, k_s = s.lower().split("x")
        shapes.append((int(n_s), int(k_s)))

    gpu = torch.cuda.get_device_name(0)
    cc = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    all_rows: list[dict] = []
    print(f"qkern {__version__} | FP16 vs INT8 vs INT4 (per-tensor fused)")
    print(f"GPU: {gpu} (sm_{cc.replace('.', '')})")
    print(
        f"{'kernel':18} {'N':>5} {'K':>6} {'lat_ms':>10} {'err':>10} "
        f"{'W_bytes':>12} {'BW_GBs':>10}"
    )
    for n, k in shapes:
        rows = bench_shape(n, k, warmup=args.warmup, iters=args.iters)
        for r in rows:
            r.update(
                {
                    "gpu": gpu,
                    "compute_capability": cc,
                    "cuda_version": torch.version.cuda,
                    "pytorch_version": torch.__version__,
                    "qkern_version": __version__,
                }
            )
            print(
                f"{r['kernel']:18} {r['N']:5d} {r['K']:6d} "
                f"{r['latency_median_ms']:10.3f} {r['max_abs_error']:10.4f} "
                f"{r['logical_weight_bytes']:12.0f} {r['logical_bandwidth_GBps']:10.3f}"
            )
            all_rows.append(r)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"fp16_int8_int4_{stamp}.json"
    csv_path = out_dir / f"fp16_int8_int4_{stamp}.csv"
    payload = {
        "meta": {
            "gpu": gpu,
            "compute_capability": cc,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "qkern_version": __version__,
            "warmup": args.warmup,
            "iters": args.iters,
            "timing": "torch.cuda.Event median",
            "experiment": "fp16_vs_int8_vs_int4_per_tensor",
            "diagnostics": diag,
        },
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
