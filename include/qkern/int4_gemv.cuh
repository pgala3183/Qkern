#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qkern {

// Fused INT4 weight-only GEMV. Does not materialize dequantized W in global memory.
//
// Packed storage (Phase-1 convention):
//   W_q[n, i] is uint8; low nibble = q[n, 2*i], high nibble = q[n, 2*i+1]
//   (high nibble may be zero padding when K is odd).
// Signed INT4: nibble in [0,15] with values >= 8 meaning -8..-1 (two's complement).
//
// Per-tensor (v1): y[n] = scale * Σ_k float(q[n,k]) * float(x[k])

void launch_int4_gemv_per_tensor_fused(
    const std::uint8_t* W_q_packed,  // [N, ceil(K/2)]
    const __half* x,                 // [K]
    float* y,                        // [N]
    float scale,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace qkern
