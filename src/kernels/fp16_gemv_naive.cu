#include "qkern/fp16_gemv.cuh"

namespace qkern {
namespace {

// Phase-2 baseline: one thread owns one output row y[n].
// Each thread walks K with scalar FP16 loads of W and x from global memory.
__global__ void fp16_gemv_naive_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    float* __restrict__ y,
    int N,
    int K) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= N) {
    return;
  }

  const __half* row = W + static_cast<std::size_t>(n) * static_cast<std::size_t>(K);
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float wv = __half2float(row[k]);
    const float xv = __half2float(x[k]);
    acc += wv * xv;
  }
  y[n] = acc;
}

}  // namespace

void launch_fp16_gemv_naive(
    const __half* W,
    const __half* x,
    float* y,
    int N,
    int K,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  fp16_gemv_naive_kernel<<<blocks, threads, 0, stream>>>(W, x, y, N, K);
}

}  // namespace qkern
