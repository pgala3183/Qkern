# QKern quantization reference

This document defines the **PyTorch reference** quantization used by QKern.
Later CUDA kernels must match this math (up to documented floating-point
accumulation differences).

## Weight-only quantization

In LLM decode, activations change every step while weights are static.
**Weight-only** quantization stores `W` in a low-bit integer format and keeps
activations in higher precision (here FP16).

For a linear layer:

```
y = W x          W ∈ R^{N×K},  x ∈ R^{K},  y ∈ R^{N}
```

the reference path is:

1. Quantize `W` offline (or once) → integer `q` + scales
2. Dequantize to a floating `Ŵ`
3. Compute `y = Ŵ x` (unfused reference)

Fused CUDA kernels will fold step 2 into the matvec; they must remain numerically
faithful to this reference.

## Symmetric quantization

QKern uses **symmetric** uniform quantization with **zero-point = 0**:

```
q = clip(round(W / scale), qmin, qmax)
Ŵ = q · scale
```

| Format | `qmin` | `qmax` | scale formula |
|--------|--------|--------|----------------|
| INT8   | -128   | 127    | `scale = max_abs / 127` |
| INT4   | -8     | 7      | `scale = max_abs / 7` |

Notes:

- **Rounding:** banker's? No — PyTorch `torch.round` (half away from even ties follow
  PyTorch semantics; we rely on `torch.round` for reproducibility).
- **Clipping:** after rounding, values outside `[qmin, qmax]` are clamped.
- **Zero tensors:** if `max_abs == 0`, scale is set to `1.0` so dequantization is
  well-defined and yields zeros.
- INT8 uses `127` (not `128`) in the denominator so the positive peak maps to
  `+127`. Representable negatives extend to `-128`, which is the standard
  asymmetric integer range for 8-bit two's complement.
- INT4 uses the project-mandated range `[-8, 7]`.

## Granularity

API:

```python
quantize_int8(W, granularity="tensor")
quantize_int8(W, granularity="channel")
quantize_int8(W, granularity="group", group_size=128)

quantize_int4(W, granularity="group", group_size=128, pack=True)
```

### Per-tensor

One scale for the entire matrix `W`.

- `max_abs = max(|W|)`
- `scales` shape: scalar `[]`

### Per-channel

One scale per **output row** (channel) of `W ∈ [N, K]`.

- `max_abs[n] = max_k |W[n, k]|`
- `scales` shape: `[N, 1]`

This matches the usual “per output feature” convention for `y = Wx`.

### Per-group

Along `K`, split each row into groups of `group_size` (default candidates:
32, 64, 128, 256). The **primary target** is **group size 128**.

- `num_groups = ceil(K / group_size)`
- For group `g`, columns `[g·gs, min((g+1)·gs, K))` may be a **partial last group**
  when `K` is not divisible by `group_size`.
- `scales` shape: `[N, num_groups]`

## Dequantization

```python
Ŵ = dequantize(qw)   # FP32 by default
Ŵ = dequantize(qw, dtype=torch.float16)
```

Scales are expanded to `[N, K]` then multiplied elementwise by the integer
weights. Packed INT4 weights are unpacked first.

## INT4 packing format

Two signed INT4 values pack into one `uint8`:

| Byte bit field | Contents |
|----------------|----------|
| bits `[3:0]` (low nibble)  | `q[..., 2i]` |
| bits `[7:4]` (high nibble) | `q[..., 2i+1]` |

Signed storage uses 4-bit two's complement nibbles:

```
stored = value & 0xF
```

so `-8 → 0x8`, `-1 → 0xF`, `0 → 0x0`, `7 → 0x7`.

If `K` is odd, a padding zero nibble is appended for packing and discarded on
unpack (`unpack_int4(packed, k=K)`).

Packed `qweight` shape: `[N, ceil(K/2)]`, dtype `uint8`.

## Error metrics

Use `qkern.metrics` for correctness reports:

- `max_abs_error`
- `mean_abs_error`
- `relative_error` — `mean(|a-b| / max(|b|, eps))`
- `rmse`

## Small numerical example (INT8, per-tensor)

```
W = [[1.5, -3.0]]
max_abs = 3.0
scale = 3.0 / 127 ≈ 0.023622
q = round(W / scale) = [[64, -127]]   (clipped/rounded)
Ŵ = q * scale ≈ [[1.5118, -3.0000]]
```

## Small numerical example (INT4 pack)

```
q = [-8, 7]  →  low=0x8, high=0x7  →  byte = 0x78
unpack → [-8, 7]
```

## Fused INT8 CUDA GEMV (implemented)

API:

```python
from qkern import quantize_int8, int8_gemv_fused_from_qw

qw = quantize_int8(W, granularity="group", group_size=128)
y = int8_gemv_fused_from_qw(qw, x.half())  # CUDA, FP32 y
```

Scale layouts consumed by the fused kernels:

| Granularity | `scales` layout | Kernel use |
|-------------|-----------------|------------|
| tensor | scalar | `y[n] = s * Σ Wq[n,k]·x[k]` |
| channel | `[N]` (also accepts `[N,1]`) | `y[n] = s[n] * Σ Wq[n,k]·x[k]` |
| group | `[N, ceil(K/gs)]` | `g=⌊k/gs⌋`, `y[n]=Σ s[n,g]·Wq[n,k]·x[k]` |

Partial last groups when `K % group_size ≠ 0` are required and tested.

### Design notes (correctness-first; not aggressively optimized)

| Topic | Behavior |
|-------|----------|
| Extra memory loads | Channel: one scale per row (amortized). Group: one scale load per `k` (or reuse if compiler keeps `s` in a register within a group — not assumed). |
| Indexing overhead | Group path does `g = k / group_size` each iteration (integer divide). |
| Cache behavior | Scales for a row are contiguous in `[num_groups]`; neighboring threads touch different rows → scale traffic is stride-`num_groups`. Weights remain row-strided across the warp (same coalescing issue as naive FP16). |
| Expected tradeoffs | Finer scales (group) usually **lower quantization error**, but **more scale traffic / indexing** than tensor. Latency is measured; no granularity is declared “best.” |

See also `docs/experiments/int8_gemv_fused.md` and `benchmarks/benchmark_int8_granularity.py`.

## Fused INT4 CUDA GEMV (implemented)

Primary QKern target: **W4A16** — weight-only INT4, FP16 activations, FP32
accumulation, **fused** dequantization (no dequantized `W` in global memory).

Primary group configuration: **group_size = 128**. Also supported: 32, 64, 256.

### Packing (Phase-1)

Two signed INT4 values in `[-8, 7]` share one `uint8` byte:

| Nibble | Bits | Logical `K` index |
|--------|------|-------------------|
| low | `[3:0]` | `2*i` |
| high | `[7:4]` | `2*i+1` (zero pad if `K` odd) |

Storage is 4-bit two's complement (`stored = value & 0xF`). Fused kernels must
match `pack_int4` / `unpack_int4`.

### API

```python
from qkern import quantize_int4, int4_gemv_fused_from_qw

qw = quantize_int4(W, granularity="group", group_size=128, pack=True)
y = int4_gemv_fused_from_qw(qw, x.half())  # CUDA, FP32 y
```

`W_q` shape is `[N, ceil(K/2)]` (`uint8`). Logical `K` comes from `qw.shape`.

Scale layouts:

| Granularity | `scales` layout | Kernel use |
|-------------|-----------------|------------|
| tensor | scalar | `y[n] = s * Σ q[n,k]·x[k]` |
| channel | `[N]` (also `[N,1]`) | `y[n] = s[n] * Σ q[n,k]·x[k]` |
| group | `[N, ceil(K/gs)]` | `g=⌊k/gs⌋`, `y[n]=Σ s[n,g]·q[n,k]·x[k]` |

Partial last groups when `K % group_size ≠ 0` are required and tested.

### Kernel steps (per loaded INT4 weight)

1. Determine group `g = floor(k / group_size)` (group path).
2. Load the corresponding scale.
3. Sign-extend the INT4 nibble.
4. Dequantize in registers (`float(q) * scale`).
5. Multiply by FP16 activation.
6. Accumulate in FP32.

Odd `K`: the final high nibble is padding and is skipped.

### Design notes (correctness-first; optimization-ready)

| Topic | Behavior |
|-------|----------|
| Extra memory loads | Channel: one scale/row. Group: scale lookup per weight (not staged yet). |
| Indexing overhead | Group path does `g = k / group_size` per weight (integer divide). |
| Cache behavior | Row scales are contiguous in `[num_groups]`; neighboring threads touch different rows. Packed weights remain row-strided across the warp. |
| Expected tradeoffs | Finer scales usually lower quantization error, more scale traffic/indexing. Latency is measured; no granularity is declared “best.” |
| Future opts (not done) | Staged scale loads, power-of-two group shifts, vectorized W/x loads, register tiling, shared-memory `x`/scales. |

### Why packing reduces memory traffic

Logical weight bytes drop from `N·K·2` (FP16) or `N·K` (INT8) to about
`N·ceil(K/2)` — roughly **4× less weight traffic than FP16** and **2× less than
INT8**, before activation/output/scale bytes.

### Unpacking overhead / instruction pressure

Nibble extract + sign-extend + per-group scale loads raise work per byte.
INT4 can become instruction-bound despite lower weight traffic — measure rather
than assume.

Compare: `benchmarks/benchmark_int4_granularity.py` and
`docs/experiments/int4_gemv_granularity.md`.

### Fused vs unfused (CUDA dequant)

Unfused path for experiments:

```python
W_hat = int4_dequant_from_qw(qw)   # CUDA → FP16 [N,K]
y = fp16_gemv(W_hat, x)            # or int4_gemv_unfused_from_qw
```

Fused path: `int4_gemv_fused_from_qw` (no `Ŵ` in global memory).

Rigorous timing of dequant / GEMV / total unfused / fused:
`benchmarks/benchmark_fused_vs_unfused.py` and
`docs/experiments/fused_vs_unfused_int4.md`. Fusion is **not** assumed to win.
