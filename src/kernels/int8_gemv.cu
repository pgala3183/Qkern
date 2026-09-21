#include "qkern/int8_gemv.cuh"

namespace qkern {
namespace {

// Intentionally simple fused path: one thread per output row.
// Per element: int8 -> float, multiply by FP16 activation (as float), FP32 accum,
// then apply the single per-tensor scale once at the end.
__global__ void int8_gemv_per_tensor_fused_kernel(
    const std::int8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    float scale,
    int N,
    int K) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= N) {
    return;
  }

  const std::int8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(K);
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float w = static_cast<float>(row[k]);
    const float xv = __half2float(x[k]);
    acc += w * xv;
  }
  y[n] = acc * scale;
}

}  // namespace

void launch_int8_gemv_per_tensor_fused(
    const std::int8_t* W_q,
    const __half* x,
    float* y,
    float scale,
    int N,
    int K,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  int8_gemv_per_tensor_fused_kernel<<<blocks, threads, 0, stream>>>(
      W_q, x, y, scale, N, K);
}

}  // namespace qkern
