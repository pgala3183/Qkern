# Experiment: fused INT4 GEMV (per-tensor v1)

## Goal

Ship a correctness-first packed INT4 fused GEMV and measure it next to FP16 and
INT8 fused paths. No winner claims without numbers.

## Method

- Kernel: one thread / output row; unpack nibbles in registers; FP32 accum.
- Quantization: per-tensor, Phase-1 packing.
- Timing: CUDA events; warmup=20, iters=100.
- Logical weight bytes: FP16 `N·K·2`, INT8 `N·K`, INT4 `N·ceil(K/2)`.
- Error: max abs vs PyTorch reference (FP16 matmul or `gemv_from_dequant`).

Command:

```powershell
.\scripts\build_ext.ps1
python benchmarks/benchmark_fp16_int8_int4.py --warmup 20 --iters 100
```

## Measured results

**Machine:** NVIDIA GeForce GTX 1650 (sm_75)  
**Stamp:** `fp16_int8_int4_20260921T100039Z`

| kernel | N | K | latency (ms) | max abs error | logical weight bytes | logical BW (GB/s) |
|--------|---|---|--------------|---------------|----------------------|-------------------|
| pytorch_fp16 | 64 | 256 | 0.058 | 0.0072 | 32768 | 0.580 |
| qkern_fp16_vec2 | 64 | 256 | 0.031 | 0.0000 | 32768 | 1.085 |
| int8_fused | 64 | 256 | 0.047 | 0.0082 | 16384 | 0.364 |
| int4_fused | 64 | 256 | 0.048 | 0.0087 | 8192 | 0.186 |
| pytorch_fp16 | 256 | 4096 | 0.088 | 0.0578 | 2097152 | 24.006 |
| qkern_fp16_vec2 | 256 | 4096 | 0.459 | 0.0002 | 2097152 | 4.592 |
| int8_fused | 256 | 4096 | 0.868 | 0.0402 | 1048576 | 1.218 |
| int4_fused | 256 | 4096 | 0.483 | 0.0247 | 524288 | 1.104 |
| pytorch_fp16 | 1024 | 4096 | 0.093 | 0.0602 | 8388608 | 90.138 |
| qkern_fp16_vec2 | 1024 | 4096 | 0.475 | 0.0003 | 8388608 | 17.692 |
| int8_fused | 1024 | 4096 | 0.845 | 0.0526 | 4194304 | 4.979 |
| int4_fused | 1024 | 4096 | 0.475 | 0.0538 | 2097152 | 4.445 |

### Observed tradeoffs (descriptive only)

- Packed INT4 halves logical weight bytes vs INT8 and quarters vs FP16.
- On these shapes with correctness-first kernels, INT4 fused latency is close to
  qkern FP16 `vec2` and often lower than the INT8 fused path — still far from
  cuBLAS-backed PyTorch FP16 on large shapes.
- Quantization error for INT4/INT8 is in the same ballpark here on random
  Gaussian weights; real LLM weights may differ.
- Lower weight traffic does not by itself prove a bandwidth win: unpacking adds
  instruction work (see `docs/kernel_design.md`).
