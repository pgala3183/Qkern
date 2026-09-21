# Experiment: INT8 fused GEMV granularities

## Scope

Extend fused INT8 weight-only GEMV beyond per-tensor:

1. **per-tensor** — scalar scale
2. **per-channel** — `scales[N]`
3. **per-group** — `scales[N, ceil(K/group_size)]` with `group_size ∈ {32,64,128,256}`

Primary group size: **128**.

Correctness prioritized; scale access is not aggressively optimized.

## Measured results

**Machine:** NVIDIA GeForce GTX 1650 (sm_75)  
**Method:** CUDA events, warmup=20, iters=100  
**Command:** `python benchmarks/benchmark_int8_granularity.py --warmup 20 --iters 100`

No granularity is declared best. Measured tradeoffs:

### N=64, K=256

| granularity | group size | latency (ms) | max abs error | logical BW (GB/s) |
|-------------|------------|--------------|---------------|-------------------|
| tensor | — | 0.078 | 0.0112 | 0.220 |
| channel | — | 0.102 | 0.0096 | 0.170 |
| group | 32 | 0.090 | 0.0110 | 0.213 |
| group | 64 | 0.100 | 0.0092 | 0.181 |
| group | 128 | 0.097 | 0.0115 | 0.181 |
| group | 256 | 0.091 | 0.0096 | 0.191 |

### N=256, K=4096

| granularity | group size | latency (ms) | max abs error | logical BW (GB/s) |
|-------------|------------|--------------|---------------|-------------------|
| tensor | — | 1.278 | 0.0419 | 0.828 |
| channel | — | 1.319 | 0.0393 | 0.803 |
| group | 32 | 2.204 | 0.0409 | 0.539 |
| group | 64 | 2.150 | 0.0376 | 0.522 |
| group | 128 | 2.152 | 0.0391 | 0.507 |
| group | 256 | 1.745 | 0.0308 | 0.616 |

### N=1024, K=4096

| granularity | group size | latency (ms) | max abs error | logical BW (GB/s) |
|-------------|------------|--------------|---------------|-------------------|
| tensor | — | 2.711 | 0.0425 | 1.552 |
| channel | — | 2.732 | 0.0512 | 1.541 |
| group | 32 | 3.996 | 0.0468 | 1.184 |
| group | 64 | 4.004 | 0.0418 | 1.116 |
| group | 128 | 4.032 | 0.0465 | 1.076 |
| group | 256 | 3.077 | 0.0431 | 1.388 |

### Observed tradeoffs (descriptive only)

- On larger shapes, **group** paths are slower than tensor/channel with this
  correctness-first kernel (per-`k` scale indexing + more scale traffic).
- Error differences across granularities are small here on random Gaussian
  weights; real LLM weight distributions may differ.
- Primary target group size **128** is correct and measured; it is not the
  lowest latency on this GPU with the current kernel.
