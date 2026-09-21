# Experiment: FP16 GEMV activation caching (`x_smem`)

Status: **measured** (see results below). One optimization only.

## 1. Current bottleneck (naive kernel)

The Phase 2 naive kernel maps **one thread → one output row** `y[n]`.
Each thread scans `k = 0 .. K-1` and reads:

- `W[n, k]` (row-contiguous for that thread)
- `x[k]` from **global memory every iteration**

### What that implies for memory traffic

| Stream | Naive behavior | Implied logical traffic |
|--------|----------------|-------------------------|
| Weights `W` | Each element read once | `N · K · 2` bytes |
| Activations `x` | **Every thread** re-reads the full vector | `N · K · 2` bytes |
| Output `y` | One store per row | `N · 4` bytes |

So activation traffic is amplified by ~`N` relative to the algorithmic minimum
`K · 2` bytes.

Hardware consequence (hypothesis before this experiment):

- Extra global loads / L2 pressure for `x`
- Especially costly when `N` is moderate and `K` is large (many threads each
  streaming the same `x`)

A second issue (uncoalesced `W` across a warp at fixed `k`) remains **out of
scope** for this change and is reserved for a later experiment.

## 2. Chosen optimization (only one)

**Activation caching in shared memory** (`fp16_gemv_x_smem`).

Not done in this change: vectorized loads, multi-output-per-thread, warp
reductions, register tiling, coalesced `W` remapping.

### Expected hardware effect

1. Threads in a block cooperatively stage a tile of `x` into shared memory.
2. All row-threads reuse that tile from SMEM instead of re-fetching `x` from
   global memory.
3. For large `K`, `x` is streamed in tiles sized from **runtime**
   `cudaDeviceProp.sharedMemPerBlock` (not a hardcoded SM table).

### Metric that should improve

- **Primary:** median CUDA-event latency vs `fp16_gemv_naive` under identical
  warmup/iters/shapes.
- **Accounting:** kernel-implied **logical bytes** for `x` drops from
  `N·K·2` to `K·2` (plus tile staging counted as the single global read of `x`).
- **Derived:** logical bandwidth = logical_bytes / median_latency.

If latency does not improve, that is a valid result (e.g. L2 already cached `x`,
or `W` coalescing dominates).

## 3. Logical byte model (used in the benchmark)

```
logical_bytes_naive  = N*K*2 + N*K*2 + N*4
logical_bytes_x_smem = N*K*2 + K*2   + N*4
logical_bw_GBs       = logical_bytes / (median_latency_s) / 1e9
```

This is an **accounting** model for decode-style analysis, not a substitute for
Nsight Compute dram counters.

## 4. Implementation notes

- Preserve `fp16_gemv_naive` unchanged in behavior.
- New launcher: `launch_fp16_gemv_x_smem`.
- Python: `fp16_gemv(..., variant=...)`, `fp16_gemv_naive`, `fp16_gemv_x_smem`.

## 5. Measured results

**Machine:** NVIDIA GeForce GTX 1650 (sm_75), PyTorch 2.11.0+cu128  
**Method:** CUDA events, warmup=20, iters=100, identical inputs per shape  
**Command:** `python benchmarks/benchmark_fp16_gemv_opt.py --warmup 20 --iters 100`

| N×K | naive median (ms) | x_smem median (ms) | speedup | logical bytes naive | logical bytes x_smem | logical BW naive (GB/s) | logical BW x_smem (GB/s) |
|-----|-------------------|--------------------|---------|---------------------|----------------------|-------------------------|--------------------------|
| 64×256 | 0.0489 | 0.0279 | **1.75×** | 65792 | 33536 | 1.35 | 1.20 |
| 256×4096 | 0.846 | 0.825 | 1.03× | 4.20e6 | 2.11e6 | 4.96 | 2.55 |
| 1024×4096 | 0.850 | 0.835 | 1.02× | 1.68e7 | 8.40e6 | 19.7 | 10.1 |
| 4096×4096 | 2.423 | 2.380 | 1.02× | 6.71e7 | 3.36e7 | 27.7 | 14.1 |

### Interpretation (honest)

- **Clear win on 64×256** (~1.75×): activation reuse matters when the working set is
  small enough that cutting redundant `x` traffic shows up in latency.
- **~1.02–1.03× on large K shapes:** latency barely moves. That is consistent with
  **`W` traffic / coalescing dominating** once `N·K` is large; removing redundant
  `x` loads does not fix strided `W` warp loads.
- Logical bandwidth for `x_smem` **drops** on large shapes because the accounting
  model halves (or more) the counted bytes while wall time stays almost flat.
  That is expected for this metric definition; it is not a hardware-counter
  measurement.

**Conclusion:** keep `x_smem` as a real, measured micro-optimization with
shape-dependent benefit. Next experiment should target **coalesced `W` access**
(still one change at a time), not stack more features onto this kernel.
