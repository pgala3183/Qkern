#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qkern {

// Unpack + dequantize packed INT4 weights into a dense FP16 matrix.
// Does not run GEMV. Used for the unfused path so dequant and GEMV can be timed
// separately.
//
// Phase-1 packing: low nibble = q[..., 2*i], high = q[..., 2*i+1].
// Output W_hat[n,k] = float(q[n,k]) * scale_(n,k) stored as __half.

void launch_int4_dequant_per_tensor(
    const std::uint8_t* W_q_packed,  // [N, ceil(K/2)]
    __half* W_hat,                   // [N, K]
    float scale,
    int N,
    int K,
    cudaStream_t stream = nullptr);

void launch_int4_dequant_per_channel(
    const std::uint8_t* W_q_packed,
    __half* W_hat,
    const float* scales,  // [N]
    int N,
    int K,
    cudaStream_t stream = nullptr);

void launch_int4_dequant_per_group(
    const std::uint8_t* W_q_packed,
    __half* W_hat,
    const float* scales,  // [N, num_groups]
    int N,
    int K,
    int group_size,
    cudaStream_t stream = nullptr);

}  // namespace qkern
