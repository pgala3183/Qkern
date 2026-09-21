# INT4 GEMV compile-time configuration

## Mapping

Decode GEMV: `y[N] = W[N,K] @ x[K]` with batch **M = 1**.

The fused INT4 kernel uses **one thread per output row**. A CUDA block owns
`BLOCK_N` consecutive rows (`blockDim.x = BLOCK_N`).

| Parameter | Role in this mapping |
|-----------|----------------------|
| **BLOCK_M** | Fixed to **1**. There is no batch/`M` tile to choose; exposing it as a free parameter would be meaningless. Documented for GEMM-style naming consistency only. |
| **BLOCK_N** | Threads per block / output rows per block. Changes grid shape, occupancy, and how many rows share an `x` tile. |
| **BLOCK_K** | Length of the activation tile staged in shared memory. `0` disables smem tiling (stream `x` from global). Must be even when > 0 (INT4 pairs). |
| **VEC_SIZE** | Packed INT4 **bytes** consumed per inner-loop step (`1`, `2`, or `4`). Tails always use scalar bytes. Affects instruction mix / load width, not math. |
| **NUM_STAGES** | Number of smem buffers for `x` tiles when `BLOCK_K > 0`. Raises smem footprint; reserved for software pipelining (current kernels rotate buffers with barriers). Ignored when `BLOCK_K == 0`. |

## Instantiated configs

| Name | BLOCK_M | BLOCK_N | BLOCK_K | VEC_SIZE | NUM_STAGES | Notes |
|------|---------|---------|---------|----------|------------|-------|
| `default` | 1 | 256 | 0 | 1 | 1 | **Preserved** original path (no smem, scalar packed loads) |
| `bn128_bk128_v1_s1` | 1 | 128 | 128 | 1 | 1 | Smaller blocks + `x` smem tiles |
| `bn256_bk256_v4_s2` | 1 | 256 | 256 | 4 | 2 | Wider vec loads + dual smem stages |

Math is identical across configs (same unpack, scales, FP32 accum). Only the
schedule / memory hierarchy path changes.

## How parameters affect resources

### Memory traffic

- **BLOCK_K = 0:** each thread re-reads `x` from global (amplified by active rows).
- **BLOCK_K > 0:** block cooperatively loads each `x` tile once into smem; threads reuse it. Logical global `x` traffic drops toward ~`ceil(K/BLOCK_K)` tile loads per block instead of per-thread full scans.
- **VEC_SIZE:** does not change logical weight bytes; may change transactions if loads coalesce better.
- **NUM_STAGES:** does not change steady-state logical traffic; enables overlapping later.

### Shared memory

- Bytes ≈ `NUM_STAGES * BLOCK_K * sizeof(half)` when `BLOCK_K > 0`, else `0`.
- Larger `BLOCK_K` / `NUM_STAGES` → more smem per block → fewer concurrent blocks.

### Registers

- Accumulators and unpack temporaries are similar across configs.
- Higher `VEC_SIZE` unrolls more live values per step → potential register pressure.
- Smem paths keep tile indices / stage pointers in registers.

### Occupancy

- **BLOCK_N** sets threads/block (and thus warps/block).
- Smem and registers per block limit blocks/SM; large `BLOCK_K * NUM_STAGES` lowers occupancy.
- `default` (0 smem) typically allows higher occupancy than tiled configs on small SMs.

### Parallelism

- Grid: `ceil(N / BLOCK_N)` blocks.
- Smaller `BLOCK_N` → more blocks (more SM-level parallelism) but more launch/tail overhead on tiny `N`.
- Larger `BLOCK_N` → fewer blocks, better amortization of `x` tile loads within a block.

## API

```python
from qkern import int4_gemv_fused_from_qw, int4_gemv_list_configs

print(int4_gemv_list_configs())
y = int4_gemv_fused_from_qw(qw, x, config="default")
y2 = int4_gemv_fused_from_qw(qw, x, config="bn128_bk128_v1_s1")
```

NVRTC / runtime compilation is **out of scope** for this phase; configs are
explicit template instantiations selected by name.

## Measured snapshot (GTX 1650, N=1024, K=4096, group=128)

CUDA events, warmup=10, iters=50. Configuration stored on every row.
Does not rank configs.

| config | BLOCK_N | BLOCK_K | VEC_SIZE | NUM_STAGES | latency (ms) | max abs error |
|--------|---------|---------|----------|------------|--------------|---------------|
| default | 256 | 0 | 1 | 1 | 14.55 | 0.0464 |
| bn128_bk128_v1_s1 | 128 | 128 | 1 | 1 | 14.07 | 0.0466 |
| bn256_bk256_v4_s2 | 256 | 256 | 4 | 2 | 14.59 | 0.0466 |
