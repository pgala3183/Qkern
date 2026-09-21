# Experiment: INT8 weight-only fused GEMV (per-tensor)

## Goal

Compute

```
y = dequant(W_int8) @ x
```

**without** writing a full dequantized `Ŵ ∈ R^{N×K}` to global memory.

## Why fusion should reduce traffic

Unfused path:

1. Read INT8 `W_q` (`N·K` bytes)
2. Write FP16 `Ŵ` (`2·N·K` bytes)
3. Read FP16 `Ŵ` again in the GEMV (`2·N·K` bytes)
4. Read `x`, write `y`

Fused path:

1. Read INT8 `W_q` once (`N·K` bytes)
2. Convert → scale → multiply by `x[k]` in registers / accumulators
3. Read `x`, write `y`

So fusion removes the **materialized dequant buffer** (write + re-read of
`~2·N·K` bytes). Whether wall-clock latency improves depends on whether that
traffic (and kernel launch / occupancy) was the bottleneck versus compute and
activation traffic.

## Scope (this phase)

- **Per-tensor** scale only
- Symmetric INT8 matching Phase-1 `quantize_int8(..., granularity="tensor")`
- FP16 activations, FP32 accumulation
- No per-channel / per-group yet

## Kernel math

```
acc[n] = Σ_k float(W_q[n,k]) * float(x[k])
y[n]   = scale * acc[n]
```

One thread owns one output row (same simple mapping as the early FP16 baseline).

## API

```python
from qkern import quantize_int8, int8_gemv_fused, int8_gemv_unfused

qw = quantize_int8(W, granularity="tensor")
y_fused = int8_gemv_fused(qw.qweight.cuda(), x.cuda().half(), float(qw.scales))
y_unfused = int8_gemv_unfused(qw.qweight.cuda(), x.cuda().half(), float(qw.scales))
```

## Measured results

**Machine:** NVIDIA GeForce GTX 1650 (sm_75), PyTorch 2.11.0+cu128  
**Method:** CUDA events, warmup=20, iters=100  
**Command:** `python benchmarks/benchmark_int8.py --warmup 20 --iters 100`

Median latency (ms). Speedup is vs **PyTorch FP16** (`>1` means faster than PyTorch).

| N×K | PyTorch FP16 | qkern FP16 vec2 | INT8 unfused | INT8 fused | fused speedup vs PyTorch | fused vs unfused |
|-----|--------------|-----------------|--------------|------------|--------------------------|------------------|
| 64×256 | 0.051 | 0.029 | 0.117 | **0.043** | **1.19×** | fused wins |
| 256×4096 | **0.068** | 0.463 | 0.620 | 0.841 | 0.08× | unfused faster than this fused kernel |
| 1024×4096 | **0.093** | 0.473 | 1.048 | 0.835 | 0.11× | fused beats unfused |
| 4096×4096 | 2.246 | **1.241** | 3.498 | 2.408 | 0.93× | fused beats unfused |

Logical weight bytes (accounting): FP16 `2NK`, fused INT8 `NK`, unfused INT8 `NK + 2NK` (source + materialized FP16).

### Interpretation (honest)

- **Fusion vs unfused:** On larger shapes, fused is faster than unfused because it avoids
  allocating/writing/re-reading a full FP16 `Ŵ`. That matches the traffic argument.
- **INT8 fused vs PyTorch FP16:** Not a consistent win on this GPU with this simple
  one-thread-per-row fused kernel. Highly tuned cuBLAS FP16 GEMV remains hard to beat
  on mid sizes; do **not** claim INT8 must be faster.
- **256×4096 anomaly:** fused slower than unfused here — likely because the fused kernel
  is still a scalar INT8 walk, while unfused pays dequant once then rides the faster
  `vec2` FP16 kernel. A future INT8 optimization (vectorized int8 loads, etc.) is
  separate work.

**Conclusion:** Per-tensor fused INT8 is correct and demonstrates the fusion traffic
win vs unfused on several shapes. Performance vs FP16 baselines is measured, not assumed.
