# AGENTS.md — working rules for QKern

This file guides automated agents and collaborators working on QKern.

## Mission

Build a serious architecture-aware quantized CUDA kernel + autotuner project for
decode-like LLM linear layers. Prefer interview-explainable handwritten CUDA
over library wrappers.

## Hard rules

1. **Correctness before optimization.** Every CUDA path needs a PyTorch reference check.
2. **Measurable optimizations only.** State bottleneck, expected effect, and measurement plan.
3. **No fabricated results.** Distinguish measured vs theoretical vs hypothesis.
4. **Do not replace core kernels** with PyTorch/Triton/CUTLASS unless the user asks. CUTLASS may be an external baseline later.
5. **Query device properties at runtime.** Never hardcode SM limits or shared-memory caps.
6. **Preserve reference implementations** when adding optimized variants.
7. **Phase discipline.** Implement only the requested phase. Do not jump ahead.
8. **Push after each completed user prompt** to `https://github.com/pgala3183/Qkern.git` once work is verified.
9. **Record benchmark metadata:** GPU, compute capability, CUDA version, PyTorch version, kernel, M/N/K, dtype, quantization, group size, config, warmup/measure counts, median/min/p25/p75 latency.

## Decode-focused metrics

For GEMV-like work report latency, logical bytes, effective logical bandwidth,
and speedup vs baseline. FLOPS are secondary.

## Style

- Prefer clear, interviewable code over clever abstractions.
- Avoid placeholder files that do nothing.
- Prefer reproducibility over cleverness.
