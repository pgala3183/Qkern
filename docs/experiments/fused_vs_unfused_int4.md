# Experiment: fused vs unfused INT4 GEMV (W4A16, group 128)

## Paths

**Unfused**

```
INT4 packed weights + scales
  → CUDA int4_dequant  →  FP16 Ŵ [N,K] in global memory
  → CUDA FP16 GEMV (vec2)
  → FP32 y
```

**Fused**

```
INT4 packed weights + scales + FP16 x
  → single CUDA int4_gemv_fused (group)
  → FP32 y
```

No intermediate dense weight matrix.

## Method

- `M=1` (decode-like GEMV)
- Shapes: `4096×4096`, `11008×4096`, `4096×11008`, `14336×4096`
- Quantization: INT4 per-group, `group_size=128`
- Timing: `torch.cuda.Event`, synchronize after every sample
- Warm up both end-to-end paths before stage measurements
- Identical event+sync methodology for dequant, GEMV-only, unfused total, fused

Logical bytes (lower bound):

| Path | Traffic counted |
|------|-----------------|
| Fused | packed INT4 + scales + `x` + `y` |
| Unfused | packed + scales + **write** `Ŵ` + **read** `Ŵ` + `x` + `y` |

Speedup = `unfused_total_latency / fused_latency` (values `< 1` mean fused was slower).

```powershell
.\scripts\build_ext.ps1
python benchmarks/benchmark_fused_vs_unfused.py --warmup 10 --iters 50
```

## Measured results

**Machine:** NVIDIA GeForce GTX 1650 (sm_75)  
**Stamp:** `fused_vs_unfused_int4_20260921T101008Z`  
**Method:** CUDA events, warmup=10, iters=50, group_size=128

| N | K | dequant (ms) | GEMV (ms) | unfused total (ms) | fused (ms) | speedup (unf/fus) | extra logical bytes |
|---|---|--------------|-----------|--------------------|------------|-------------------|---------------------|
| 4096 | 4096 | 0.664 | 1.249 | 1.904 | 3.698 | 0.515 | 67,108,864 |
| 11008 | 4096 | 1.767 | 2.846 | 4.833 | 10.693 | 0.452 | 180,355,072 |
| 4096 | 11008 | 1.754 | 3.427 | 5.379 | 7.050 | 0.763 | 180,355,072 |
| 14336 | 4096 | 2.296 | 3.302 | 5.842 | 13.549 | 0.431 | 234,881,024 |

Machine-readable: `benchmarks/results/fused_vs_unfused_int4_20260921T101008Z.json` (and `.csv`).

**Observation:** On this GPU with the current kernels, **unfused was faster on every shape** (speedup < 1). Fusion reduced logical bytes but increased latency. That is a measured result, not a claim that fusion cannot win after optimization.

## Technical analysis

### Additional global-memory traffic (unfused)

Unfused must **write** a dense FP16 `Ŵ` (`N·K·2` bytes) then **read it again** in GEMV. Relative to fused, that is roughly an extra `2·N·K·2` bytes of weight-side traffic on top of still reading packed INT4 + scales for dequant. Those extra tens–hundreds of MB (see table) are real DRAM/cache pressure that fusion avoids in principle.

### Kernel launch overhead

Unfused issues **two** launches (dequant + GEMV) vs **one** fused launch. On these large shapes, launch overhead is small next to kernel work: dequant+GEMV stage sums nearly match end-to-end unfused totals. Launch cost would matter more for tiny `N`/`K` or batched micro-ops.

### How fusion changes the memory-access pattern

- **Unfused dequant** walks packed bytes and streams contiguous FP16 stores into `Ŵ` (write-friendly, coalesced along `K` within a row of packed indices).
- **Unfused GEMV (`vec2`)** then reads dense FP16 with safe `__half2` loads — higher bytes/op efficiency than scalar INT4 unpack.
- **Fused** keeps packed footprint but does unpack + group-scale lookup + FMA **in the same pass**, with the current correctness-first mapping (one thread per output row, scalar loads, `k/gs` per weight). That trades fewer logical bytes for **more instructions per useful byte** and a less optimized load path than `vec2`.

### Cases where fusion helps less than expected (as measured here)

1. **Unoptimized fused kernel vs optimized FP16 GEMV stage** — fusion’s bandwidth advantage is theoretical until the fused kernel’s arithmetic/unpack path is competitive.
2. **Dequant is relatively cheap** — streaming unpack+store was ~0.7–2.3 ms here; the fused kernel spent more time overall than dequant+`vec2` combined.
3. **Extra traffic can be hidden** — once `Ŵ` is resident, the GEMV may hitch on L2/DRAM bandwidth more effectively than a still-scalar fused inner loop.
4. **Shape / aspect ratio** — tall thin (`N≫K`) amplifies the cost of a slow per-row fused loop; wider `K` (4096×11008) narrowed the gap (speedup 0.76) but still did not favor fusion.
5. **Correctness-first group scales** — per-weight scale loads and integer divides in the fused path add work that unfused pays once during dequant, then GEMV sees plain FP16.

### Takeaway

Fusion is motivated by eliminating `Ŵ` traffic and one kernel boundary, but **latency wins require a fused kernel that is at least competitive with “dequant + strong FP16 GEMV.”** These measurements show the gap that later optimizations (vectorized packed loads, staged scales, activation smem, better thread mapping) need to close. Do not assume fusion always wins.
