#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qkern {

// Fused INT4 weight-only GEMV (W4A16). Does not materialize dequantized W.
//
// Packed storage (Phase-1):
//   W_q[n, i] uint8; low nibble = q[n, 2*i], high = q[n, 2*i+1]
//   (high nibble may be zero padding when K is odd).
//
// Per-tensor:  y[n] = scale * Σ_k float(q[n,k]) * float(x[k])
// Per-channel: y[n] = scales[n] * Σ_k float(q[n,k]) * float(x[k])
// Per-group:   y[n] = Σ_k scales[n, floor(k/gs)] * float(q[n,k]) * float(x[k])
//
// Kernels are correctness-first (1 thread / row). Structure leaves hooks for
// later: staged scale loads, cheaper group indexing, vectorized W/x loads,
// register tiling, and shared-memory reuse of x / scales.

void launch_int4_gemv_per_tensor_fused(
    const std::uint8_t* W_q_packed,  // [N, ceil(K/2)]
    const __half* x,                 // [K]
    float* y,                        // [N]
    float scale,
    int N,
    int K,
    cudaStream_t stream = nullptr);

void launch_int4_gemv_per_channel_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    const float* scales,  // [N]
    int N,
    int K,
    cudaStream_t stream = nullptr);

void launch_int4_gemv_per_group_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    const float* scales,  // [N, num_groups] row-major
    int N,
    int K,
    int group_size,
    cudaStream_t stream = nullptr);

}  // namespace qkern
