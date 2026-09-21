#include "qkern/int4_gemv.cuh"

namespace qkern {
namespace {

// Sign-extend a 4-bit two's-complement nibble to signed int in [-8, 7].
__device__ __forceinline__ int sign_extend_nibble(unsigned int nibble) {
  const int v = static_cast<int>(nibble & 0xFu);
  return (v >= 8) ? (v - 16) : v;
}

// Correctness-first fused INT4 GEMV: one thread per output row.
// Unpacks two INT4 values per byte in registers; no dequant buffer in global memory.
__global__ void int4_gemv_per_tensor_fused_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    float scale,
    int N,
    int K,
    int packed_k) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= N) {
    return;
  }

  const std::uint8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k);

  float acc = 0.0f;
  for (int i = 0; i < packed_k; ++i) {
    const unsigned int p = static_cast<unsigned int>(row[i]);
    const int q_lo = sign_extend_nibble(p);         // k = 2*i
    const int q_hi = sign_extend_nibble(p >> 4);    // k = 2*i + 1

    const int k0 = i * 2;
    acc += static_cast<float>(q_lo) * __half2float(x[k0]);
    if (k0 + 1 < K) {
      // Skip padding high nibble when K is odd.
      acc += static_cast<float>(q_hi) * __half2float(x[k0 + 1]);
    }
  }
  y[n] = acc * scale;
}

}  // namespace

void launch_int4_gemv_per_tensor_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    float scale,
    int N,
    int K,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  const int packed_k = (K + 1) / 2;
  constexpr int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  int4_gemv_per_tensor_fused_kernel<<<blocks, threads, 0, stream>>>(
      W_q_packed, x, y, scale, N, K, packed_k);
}

}  // namespace qkern
