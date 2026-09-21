# Experiment: INT4 fused GEMV granularities (W4A16)

## Goal

Measure per-tensor, per-channel, and per-group (32/64/128/256) INT4 fused GEMV.
Primary target is **group_size=128**. Do not rank configurations.

## Method

- Kernel: one thread / row; unpack nibbles; group scale per weight (correctness-first).
- Timing: CUDA events; warmup=20, iters=100.
- Logical bytes: packed INT4 + scales + `x` + `y`.

```powershell
.\scripts\build_ext.ps1
python benchmarks/benchmark_int4_granularity.py --warmup 20 --iters 100
```

## Measured results

**Machine:** NVIDIA GeForce GTX 1650 (sm_75)  
**Stamp:** `int4_gemv_granularity_20260921T100512Z`

### N=64, K=256

| granularity | group size | latency (ms) | max abs error | logical BW (GB/s) |
|-------------|------------|--------------|---------------|-------------------|
| tensor | — | 0.078 | 0.0084 | 0.114 |
| channel | — | 0.096 | 0.0076 | 0.096 |
| group | 32 | 0.111 | 0.0091 | 0.100 |
| group | 64 | 0.108 | 0.0108 | 0.093 |
| group | 128 | 0.130 | 0.0092 | 0.073 |
| group | 256 | 0.101 | 0.0076 | 0.091 |

### N=256, K=4096

| granularity | group size | latency (ms) | max abs error | logical BW (GB/s) |
|-------------|------------|--------------|---------------|-------------------|
| tensor | — | 0.709 | 0.0309 | 0.753 |
| channel | — | 0.730 | 0.0444 | 0.732 |
| group | 32 | 1.649 | 0.0441 | 0.403 |
| group | 64 | 1.578 | 0.0415 | 0.380 |
| group | 128 | 1.547 | 0.0464 | 0.366 |
| group | 256 | 1.142 | 0.0410 | 0.482 |

### N=1024, K=4096

| granularity | group size | latency (ms) | max abs error | logical BW (GB/s) |
|-------------|------------|--------------|---------------|-------------------|
| tensor | — | 1.381 | 0.0282 | 1.527 |
| channel | — | 1.354 | 0.0714 | 1.561 |
| group | 32 | 2.413 | 0.0483 | 1.091 |
| group | 64 | 2.257 | 0.0445 | 1.051 |
| group | 128 | 2.219 | 0.0464 | 1.010 |
| group | 256 | 1.874 | 0.0504 | 1.161 |

### Observed tradeoffs (descriptive only)

- On larger shapes, **group** paths are slower than tensor/channel with this
  correctness-first kernel (per-weight scale load + `k/gs` indexing).
- Among group sizes, **smaller groups** tend to cost more latency here (more
  distinct scales / more scale traffic); **256** is closer to channel latency
  than **32**.
- Quantization error differences across granularities are modest on random
  Gaussian weights; real LLM weight distributions may amplify the benefit of
  finer scales.
- Primary target **group_size=128** is correct and measured; it is not the
  lowest-latency config on this GPU with the current kernel.
- No configuration is declared best — latency, error, and bandwidth trade off.
