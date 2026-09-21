# Experiment: FP16 GEMV vectorized loads (`vec2`)

Status: **measured** (see §5). One optimization only — vectorized loads.

## 1. Inspection of current kernels

### `fp16_gemv_naive`
- One thread → one row `y[n]`
- Loop `k = 0 .. K-1` with **scalar** `__half` loads of `W[n,k]` and `x[k]`

### `fp16_gemv_x_smem`
- Same thread mapping
- Scalar loads of `x` into shared memory, then scalar `W` / `x_tile` products

### Which loads can be vectorized?

Along a row, `W[n, k]` and `W[n, k+1]` are adjacent in memory (row-major).
Similarly `x[k]` and `x[k+1]` are adjacent. Those pairs are candidates for
**`__half2` (32-bit) vector loads**.

Not vectorized in this experiment: cross-row / warp-strided `W` pattern
(that needs a different mapping, not wider loads).

## 2. Alignment requirements

A `__half2` load requires the address to be **4-byte aligned**.

| Pointer | Typical case | Risk |
|---------|--------------|------|
| `x` base | CUDA allocator → often 16B-aligned | May still be only 2B-aligned if a view |
| `W` row `n` | `W + n*K` | If `W` is 4B-aligned and **`K` is odd**, odd rows are only **2B-aligned** |

We **do not assume** alignment. The kernel checks addresses at runtime.

## 3. Non-divisible / misaligned `K`

Vector width = 2 halves.

1. Optional **scalar prefix** until both `row+k` and `x+k` are 4B-aligned
   (needed when both start at `addr % 4 == 2`).
2. **`__half2` main loop** while `k + 1 < K` and both addresses stay 4B-aligned.
3. **Scalar epilogue** for a leftover element when `K` is odd, or the entire
   row if dual 4B alignment never holds (e.g. `W` row and `x` differ by 2B).

If dual alignment is impossible for a row, that row is entirely scalar — still
correct.

## 4. Expected hardware effect

| Effect | Expectation |
|--------|-------------|
| Instruction overhead | Fewer load / convert / loop iterations (~2× fewer loads per thread when vector path runs) |
| Memory transactions | One 32-bit load can replace two 16-bit loads for a **contiguous** pair; helps per-thread DRAM/L1 efficiency slightly |
| Coalescing across warp | **Unchanged** — warp still issues strided `W` addresses at a given `k` |

So vectorization is primarily an **instruction + per-thread load-width** play,
not a fix for the GEMV coalescing bottleneck.

## 5. Measured results

**Machine:** NVIDIA GeForce GTX 1650 (sm_75), PyTorch 2.11.0+cu128  
**Method:** CUDA events, warmup=20, iters=100  
**Command:** `python benchmarks/benchmark_fp16_gemv_vec.py --warmup 20 --iters 100`

Median latency (ms) and speedup vs **naive**:

| N×K | naive | x_smem | vec2 | vec2 speedup vs naive | vec2 vs x_smem |
|-----|-------|--------|------|----------------------|----------------|
| 64×256 | 0.0307 | 0.0274 | **0.0246** | **1.25×** | faster than x_smem |
| 256×4096 | 0.844 | 0.821 | **0.454** | **1.86×** | much faster than x_smem |
| 1024×4096 | 0.830 | 0.820 | **0.470** | **1.77×** | much faster than x_smem |
| 4096×4096 | 2.418 | 2.374 | **1.245** | **1.94×** | much faster than x_smem |
| 16×65 (odd K) | 0.080 | 0.063 | **0.034** | **2.37×** | faster than x_smem |
| 16×64 | 0.027 | 0.033 | 0.028 | 0.95× | ≈naive; x_smem slightly worse |

### Did vectorization improve performance?

**Yes, on this GPU for the measured decode-like shapes**, typically **~1.25–1.9×**
vs naive and clearly ahead of `x_smem` on large `K`.

Caveats (honest):

- Tiny shapes (e.g. 16×64) are noisy; vec2 was ~flat vs naive there.
- `x_smem` remains a different optimization (activation reuse); vec2 does **not**
  replace it architecturally — it reduces per-thread load instruction count /
  load width on the still-strided `W` walk.
- Logical-byte accounting for vec2 matches naive (`N·K·2` for `x`); the win is
  not from fewer logical bytes but from fewer / wider load instructions.

**Conclusion:** keep `vec2` as a measured win. Next optimization should still
target coalesced `W` access separately, not fold more features into `vec2` yet.
