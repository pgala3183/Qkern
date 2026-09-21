#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qkern {

// Fused INT8 weight-only GEMV. Does not materialize dequantized W in global memory.
//
// Per-tensor:  y[n] = scale * Σ_k float(W_q[n,k]) * float(x[k])
// Per-channel: y[n] = scales[n] * Σ_k float(W_q[n,k]) * float(x[k])
// Per-group:   y[n] = Σ_k scales[n, floor(k/gs)] * float(W_q[n,k]) * float(x[k])

void launch_int8_gemv_per_tensor_fused(
    const std::int8_t* W_q,
    const __half* x,
    float* y,
    float scale,
    int N,
    int K,
    cudaStream_t stream = nullptr);

void launch_int8_gemv_per_channel_fused(
    const std::int8_t* W_q,
    const __half* x,
    float* y,
    const float* scales,  // [N]
    int N,
    int K,
    cudaStream_t stream = nullptr);

void launch_int8_gemv_per_group_fused(
    const std::int8_t* W_q,
    const __half* x,
    float* y,
    const float* scales,  // [N, num_groups] row-major
    int N,
    int K,
    int group_size,
    cudaStream_t stream = nullptr);

}  // namespace qkern
