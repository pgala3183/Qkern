#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qkern {

// Fused INT8 weight-only GEMV with per-tensor scale (Phase: INT8 fused).
// y[n] = scale * sum_k float(W_q[n,k]) * float(x[k])
// Does not materialize a dequantized weight matrix in global memory.
void launch_int8_gemv_per_tensor_fused(
    const std::int8_t* W_q,
    const __half* x,
    float* y,
    float scale,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace qkern
