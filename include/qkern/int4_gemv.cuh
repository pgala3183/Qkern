#pragma once

#include "qkern/int4_gemv_config.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qkern {

// Fused INT4 weight-only GEMV (W4A16). Does not materialize dequantized W.
//
// Packed storage (Phase-1):
//   W_q[n, i] uint8; low nibble = q[n, 2*i], high = q[n, 2*i+1]
//
// Compile-time configs: see int4_gemv_config.cuh (BLOCK_N / BLOCK_K / VEC_SIZE /
// NUM_STAGES). Default config preserves the original scalar, no-smem path.
// BLOCK_M is fixed at 1 for this GEMV mapping.

void launch_int4_gemv_per_tensor_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    float scale,
    int N,
    int K,
    Int4GemvConfigId config = Int4GemvConfigId::Default,
    cudaStream_t stream = nullptr);

void launch_int4_gemv_per_channel_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    Int4GemvConfigId config = Int4GemvConfigId::Default,
    cudaStream_t stream = nullptr);

void launch_int4_gemv_per_group_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    int group_size,
    Int4GemvConfigId config = Int4GemvConfigId::Default,
    cudaStream_t stream = nullptr);

}  // namespace qkern
