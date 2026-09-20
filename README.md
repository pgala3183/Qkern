# QKern — Architecture-Aware Quantized CUDA Kernels and Autotuning

**Thesis:** How much can fused weight-only INT8/INT4 CUDA kernels improve
decode-like LLM linear-layer performance, and how does the optimal CUDA kernel
configuration change with workload shape and GPU architecture?

This is a systems / performance engineering project aimed at GPU software,
CUDA, ML systems, and LLM inference roles. Core kernels are handwritten CUDA.
PyTorch references exist for correctness; CUTLASS / external libraries may be
used later as optional baselines only.

## Primary target

- Weight-only **INT4**, FP16 activations, FP32 accumulation
- Group size **128**
- Batch-size-1 / decode-like **GEMV** regime: `y = Wx` with `W ∈ [N, K]`, `x ∈ [K]`

## Metrics (GEMV-like)

Prefer **latency**, **logical bytes moved**, **effective logical bandwidth**,
and **speedup vs baseline**. Report achieved FLOPS only where meaningful.
Distinguish logical bandwidth from Nsight Compute hardware counters.

## Engineering rules

1. Correctness before optimization.
2. Every optimization must be measurable.
3. Never claim a speedup without benchmark evidence.
4. Do not fabricate hardware results.
5. Query GPU properties at runtime; do not hardcode hardware limits.
6. Record GPU, CUDA version, shape, dtype, kernel config, warmup/measure counts,
   and timing methodology for every benchmark.

## Phase status

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | **done** | Repo scaffold, CUDA smoke, NVRTC check, Python diagnostics |
| 1+ | pending | PyTorch reference, kernels, autotuner, benchmarks |

## Measured environment (Phase 0)

Captured on the development machine used for the initial smoke run
(`.\scripts\run_phase0.ps1`). Do not treat this as a performance result.

| Item | Measured value |
|------|----------------|
| GPU | NVIDIA GeForce GTX 1650 |
| Compute capability | 7.5 |
| Global memory | 4.000 GiB |
| SMs | 14 |
| CUDA driver / runtime | 13.3 / 13.3 |
| nvcc | 13.3.73 |
| NVRTC | 13.3 (runtime compile smoke OK) |
| Python | 3.11.4 |
| PyTorch | 2.12.1+cpu (`torch.cuda.is_available() == False`) |

**Known toolchain quirks on this host**

- System `CUDA_PATH` / `CUDA_PATH_V13_3` may point at `bin` or `libnvvp`. The Phase 0 script overrides them to the real toolkit root for the build process.
- A CUDA-enabled PyTorch wheel is still required before Phase 1 reference / correctness work can exercise the GPU from Python.

## Quick start (Phase 0)

### Prerequisites

- NVIDIA GPU + driver
- CUDA Toolkit (nvcc + NVRTC)
- CMake ≥ 3.24
- MSVC Build Tools (Windows) or a C++17 toolchain
- Python ≥ 3.10

### Build smoke targets

```powershell
# From repo root (Windows). Uses VS Build Tools + CUDA Toolkit.
.\scripts\run_phase0.ps1
```

Or manually:

```powershell
$env:CUDAToolkit_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
& "C:\Program Files\CMake\bin\cmake.exe" -S . -B build -G "Visual Studio 17 2022" -A x64
& "C:\Program Files\CMake\bin\cmake.exe" --build build --config Release
.\build\Release\qkern_device_query.exe
.\build\Release\qkern_nvrtc_smoke.exe
```

### Python diagnostics

```powershell
python -m pip install -e .
python -m qkern.diagnostics
```

## Repository layout

Incremental. Empty stubs are avoided; directories grow as phases land.

## Remote

https://github.com/pgala3183/Qkern
