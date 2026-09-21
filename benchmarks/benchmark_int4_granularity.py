"""
INT4 fused GEMV granularity benchmark (W4A16).

Compares:
  - per-tensor
  - per-channel
  - per-group 32 / 64 / 128 / 256

Reports latency, quantization error, logical bandwidth.
Does not rank configurations — measured tradeoffs only.
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
    int4_gemv_fused_from_qw,
    is_cuda_extension_available,
    max_abs_error,
    quantize_int4,
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
    """Packed INT4 weights + scales + x + y (logical lower bound)."""
    w = float(n * ((k + 1) // 2))
    if granularity == "tensor":
        s = 4.0
    elif granularity == "channel":
        s = float(n * 4)
    else:
        assert group_size is not None
        num_groups = (k + group_size - 1) // group_size
        s = float(n * num_groups * 4)
    return w + s + float(k * 2 + n * 4)


def configs() -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = [("tensor", None), ("channel", None)]
    for gs in GROUP_SIZES:
        out.append(("group", gs))
    return out


def bench_shape(n: int, k: int, *, warmup: int, iters: int) -> list[dict]:
    torch.manual_seed(0)
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    rows: list[dict] = []

    for granularity, group_size in configs():
        qw = quantize_int4(W, granularity=granularity, group_size=group_size, pack=True)
        y_ref = gemv_from_dequant(qw, x.cpu().float())

        def run(qw=qw):
            return int4_gemv_fused_from_qw(qw, x)

        out = run()
        torch.cuda.synchronize()
        err = float(max_abs_error(out.cpu(), y_ref))
        samples = time_cuda(run, warmup=warmup, iters=iters)
        med = _median(samples)
        total = logical_bytes(n, k, granularity, group_size)
        med_s = med * 1e-3
        bw = total / med_s / 1e9 if med_s > 0 else float("nan")
        rows.append(
            {
                "granularity": granularity,
                "group_size": group_size if group_size is not None else "",
                "M": 1,
                "N": n,
                "K": k,
                "latency_median_ms": med,
                "latency_min_ms": float(min(samples)),
                "max_abs_error": err,
                "logical_bytes": total,
                "logical_bandwidth_GBps": bw,
                "warmup": warmup,
                "iters": iters,
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
    )
    args = p.parse_args()

    if not torch.cuda.is_available() or not is_cuda_extension_available():
        print("CUDA + qkern._C required", file=sys.stderr)
        return 1

    gpu = torch.cuda.get_device_name(0)
    cc = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    shapes = []
    for s in args.shapes:
        n_s, k_s = s.lower().split("x")
        shapes.append((int(n_s), int(k_s)))

    print(f"qkern {__version__} | INT4 granularity (W4A16)")
    print(f"GPU: {gpu} (cc {cc})")
    print(
        f"{'granularity':12} {'gs':>4} {'N':>5} {'K':>6} "
        f"{'lat_ms':>10} {'err':>10} {'BW_GBs':>10}"
    )

    all_rows: list[dict] = []
    for n, k in shapes:
        for r in bench_shape(n, k, warmup=args.warmup, iters=args.iters):
            r.update(
                {
                    "gpu": gpu,
                    "compute_capability": cc,
                    "cuda_version": torch.version.cuda,
                    "pytorch_version": torch.__version__,
                    "qkern_version": __version__,
                }
            )
            gs = r["group_size"] if r["group_size"] != "" else "-"
            print(
                f"{r['granularity']:12} {str(gs):>4} {r['N']:5d} {r['K']:6d} "
                f"{r['latency_median_ms']:10.3f} {r['max_abs_error']:10.4f} "
                f"{r['logical_bandwidth_GBps']:10.3f}"
            )
            all_rows.append(r)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"int4_gemv_granularity_{stamp}.json"
    csv_path = out_dir / f"int4_gemv_granularity_{stamp}.csv"
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
            "experiment": "int4_gemv_granularity",
            "primary": "W4A16 group_size=128",
            "diagnostics": collect_diagnostics(),
        },
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
