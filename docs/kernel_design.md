# QKern kernel design

## Phase 2 naive FP16 GEMV

### Operation

```
y = W x
W ∈ R^{N×K} (FP16, row-major)
x ∈ R^{K}     (FP16)
y ∈ R^{N}     (FP32 accumulator / output)
```

### Thread mapping

The Phase 2 kernel is intentionally the simplest useful mapping:

- **One thread → one output element** `y[n]`
- Grid: `ceil(N / 256)` blocks × **256** threads
- Thread `n = blockIdx.x * blockDim.x + threadIdx.x` exits if `n >= N`
- That thread walks `k = 0 .. K-1`, reading `W[n, k]` and `x[k]`

No warps specialize on tiles. No shared-memory staging. No vectorized loads.
No `__hfma` / warp reductions / tensor cores.

### Memory access pattern

For a fixed thread `n`:

1. `W` accesses are along row `n` — consecutive `k` are contiguous in row-major
   layout, so **a single thread’s** `W` walk is contiguous.
2. Across a warp, neighboring threads own neighboring rows. At a given `k`, they
   touch addresses `W[n,k], W[n+1,k], …` which are **stride-`K` apart**.
   That is a classic **strided / poorly coalesced** warp load pattern when `K`
   is large.
3. Every thread reloads the full vector `x` from global memory. There is **no
   reuse via shared memory**, so `x` traffic is amplified by ~`N` (more precisely
   by the number of active threads that each scan `K`).

Logical bytes (lower bound, ignoring cache hits):

```
bytes ≈ N·K·sizeof(FP16) + N·K·sizeof(FP16)  (W + per-thread x reads)
        + N·sizeof(FP32)                     (y stores)
```

Caches may reduce `x` traffic in practice; the kernel does not rely on that.

### Why this implementation is intentionally naive

Phase 2 exists to create a **correct, interview-explainable CUDA baseline**
before optimization:

| Choice | Why naive |
|--------|-----------|
| 1 thread / output | Clear ownership; easy to validate |
| Scalar FP16 loads | No `half2` / vectorized `ld.global.v2` |
| No shared memory | No staging of `x` or tiles of `W` |
| No autotuning | Fixed 256 threads/block |
| FP32 `float` accum | Matches reference math; still simple |

Optimized variants should keep this baseline and add new kernels beside it
rather than rewriting history.

### Expected bottlenecks (hypothesis — measure later)

Decode-like GEMV is typically **memory-bandwidth / memory-latency bound**, not
FLOP bound. For this naive kernel specifically, expect:

1. **Uncoalesced `W` loads across a warp** (row-strided at fixed `k`)
2. **Redundant `x` loads** (no shared-memory broadcast/reuse)
3. **Low arithmetic intensity** (`1` FMA worth of work per 4+ bytes if `x` hits
   cache; worse if not)
4. **Launch / tail effects** for tiny `N` (e.g. `N=1` uses one thread)

These are hypotheses to confirm with latency benchmarks and, later, Nsight
Compute. Phase 2 benchmarks record median CUDA-event latency only; they do
**not** claim the custom kernel is faster than cuBLAS/PyTorch.

### Python API

```python
import torch
from qkern import fp16_gemv, fp16_gemv_naive, fp16_gemv_x_smem, fp16_gemv_ref

W = torch.randn(N, K, device="cuda", dtype=torch.float16).contiguous()
x = torch.randn(K, device="cuda", dtype=torch.float16).contiguous()
y0 = fp16_gemv_naive(W, x)                 # Phase-2 baseline (preserved)
y1 = fp16_gemv(W, x, variant="x_smem")     # first optimization only
y_ref = fp16_gemv_ref(W.cpu(), x.cpu())
```

#### Variant framework

| Variant | API | Change vs naive |
|---------|-----|-----------------|
| `naive` | `fp16_gemv_naive` | Baseline (one thread/row, global `x`) |
| `x_smem` | `fp16_gemv_x_smem` | Tile `x` through shared memory only |
| `vec2` | `fp16_gemv_vec2` | Safe `__half2` loads + scalar fallback |

Experiment write-ups:
- [experiments/fp16_gemv_x_smem.md](experiments/fp16_gemv_x_smem.md)
- [experiments/fp16_gemv_vec2.md](experiments/fp16_gemv_vec2.md)

Validation rejects CPU tensors, non-FP16 dtypes, non-contiguous layouts, and
shape mismatches with clear errors.

### Measured Phase 2 latency snapshot (GTX 1650)

CUDA-event median latency (ms), warmup=10, iters=50. **Not** an optimization claim;
the naive kernel is expected to lag highly tuned cuBLAS paths on many shapes.

| N×K | PyTorch matmul | cuBLAS-backed matmul | qkern naive |
|-----|----------------|----------------------|-------------|
| 1×1 | 0.069 | 0.056 | 0.043 |
| 64×256 | 0.064 | 0.059 | 0.180 |
| 256×4096 | 0.313 | 0.315 | 9.023 |
| 1024×4096 | 0.981 | 1.018 | 8.956 |
| 4096×4096 | 25.584 | 25.864 | 26.661 |

Notes:

- PyTorch paths return FP16; qkern returns FP32 (FP32 accumulation).
- Mid-size rows show the expected naive-kernel gap (strided `W` + redundant `x` traffic).
- Full JSON/CSV: `benchmarks/results/` (local; gitignored).

---

## Fused INT4 GEMV (primary kernel)

### Operation

```
# per-tensor / channel (scale outside or once per row):
y[n] = scale * Σ_k float(q[n,k]) * float(x[k])

# per-group:
y[n] = Σ_k scales[n, floor(k/gs)] * float(q[n,k]) * float(x[k])
```

- `q` is signed INT4 in `[-8, 7]`, packed two-per-byte (Phase-1 layout).
- `x` is FP16; accumulation and `y` are FP32.
- Primary config: **W4A16, group_size=128**. Also: 32, 64, 256.
- Dequantization is **fused**.

### Thread mapping (correctness-first)

**One thread per output row**, 256 threads/block. Helpers are factored so later
passes can optimize:

| Hook | Current | Later |
|------|---------|-------|
| Scale loads | Per-weight (group) / once (channel/tensor) | Stage in smem / registers |
| Group index | `k / group_size` | Shift when `gs` is power-of-two |
| Weight loads | Scalar `uint8` | Vectorized multi-byte |
| Activation loads | Scalar `__half` | `__half2` / smem broadcast |
| Register / smem | None beyond locals | Tile K; cache `x` and row scales |

### Per-weight group path

1. Determine `g = floor(k / group_size)` (incomplete final groups OK).
2. Load `scales[n, g]`.
3. Sign-extend INT4.
4. Dequantize in registers.
5. Multiply by activation.
6. Accumulate in FP32.

API: `int4_gemv_fused` / `int4_gemv_fused_from_qw` with optional `config=`.
Compile-time parameters: [kernel_config.md](kernel_config.md).
See also [quantization.md](quantization.md) and
[experiments/int4_gemv_granularity.md](experiments/int4_gemv_granularity.md).
