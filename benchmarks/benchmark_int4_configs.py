"""
Benchmark multiple compile-time INT4 GEMV configurations on the same problem.

Stores full config (BLOCK_M/N/K, VEC_SIZE, NUM_STAGES) with every result row.
Does not claim a best config.
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
    int4_gemv_list_configs,
    is_cuda_extension_available,
    max_abs_error,
    quantize_int4,
)
from qkern.diagnostics import collect_diagnostics


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--group-size", type=int, default=128)
    args = p.parse_args()

    if not torch.cuda.is_available() or not is_cuda_extension_available():
        print("CUDA + qkern._C required", file=sys.stderr)
        return 1

    configs = int4_gemv_list_configs()
    n, k = args.n, args.k
    torch.manual_seed(0)
    W = torch.randn(n, k)
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    qw_cpu = quantize_int4(W, granularity="group", group_size=args.group_size, pack=True)
    from dataclasses import replace

    qw = replace(
        qw_cpu,
        qweight=qw_cpu.qweight.cuda().contiguous(),
        scales=qw_cpu.scales.to(device="cuda", dtype=torch.float32).contiguous(),
    )
    y_ref = gemv_from_dequant(qw_cpu, x.cpu().float())

    gpu = torch.cuda.get_device_name(0)
    cc = ".".join(str(x_) for x_ in torch.cuda.get_device_capability(0))
    print(f"qkern {__version__} | INT4 config sweep N={n} K={k} gs={args.group_size}")
    print(f"GPU: {gpu} (cc {cc})")
    print(
        f"{'config':22} {'BN':>4} {'BK':>4} {'V':>2} {'S':>2} "
        f"{'lat_ms':>10} {'err':>10}"
    )

    rows: list[dict] = []
    for cfg in configs:
        name = cfg["name"]

        def run(name=name):
            return int4_gemv_fused_from_qw(qw, x, config=name)

        out = run()
        torch.cuda.synchronize()
        err = float(max_abs_error(out.cpu(), y_ref))
        samples = time_cuda(run, warmup=args.warmup, iters=args.iters)
        med = _median(samples)
        row = {
            "config": name,
            "BLOCK_M": int(cfg["BLOCK_M"]),
            "BLOCK_N": int(cfg["BLOCK_N"]),
            "BLOCK_K": int(cfg["BLOCK_K"]),
            "VEC_SIZE": int(cfg["VEC_SIZE"]),
            "NUM_STAGES": int(cfg["NUM_STAGES"]),
            "is_default": bool(cfg["is_default"]),
            "M": 1,
            "N": n,
            "K": k,
            "group_size": args.group_size,
            "granularity": "group",
            "latency_median_ms": med,
            "latency_min_ms": float(min(samples)),
            "max_abs_error": err,
            "warmup": args.warmup,
            "iters": args.iters,
            "gpu": gpu,
            "compute_capability": cc,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "qkern_version": __version__,
        }
        print(
            f"{name:22} {row['BLOCK_N']:4d} {row['BLOCK_K']:4d} "
            f"{row['VEC_SIZE']:2d} {row['NUM_STAGES']:2d} "
            f"{med:10.3f} {err:10.4f}"
        )
        rows.append(row)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"int4_gemv_configs_{stamp}.json"
    csv_path = out_dir / f"int4_gemv_configs_{stamp}.csv"
    payload = {
        "meta": {
            "experiment": "int4_gemv_compile_time_configs",
            "gpu": gpu,
            "compute_capability": cc,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "qkern_version": __version__,
            "warmup": args.warmup,
            "iters": args.iters,
            "timing": "torch.cuda.Event median",
            "note": "Does not rank configs. Configuration stored per row.",
            "diagnostics": collect_diagnostics(),
            "available_configs": configs,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
