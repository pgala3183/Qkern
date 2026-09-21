#include "qkern/fp16_gemv.cuh"

#include <cstdint>

namespace qkern {
namespace {

__device__ __forceinline__ bool is_half2_aligned(const __half* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 0x3u) == 0u;
}

// Scalar fallback path (same math as naive). Used when vectorization cannot run
// or for prefix/epilogue elements.
__device__ __forceinline__ float fp16_row_dot_scalar(
    const __half* __restrict__ row,
    const __half* __restrict__ x,
    int k_begin,
    int K) {
  float acc = 0.0f;
  for (int k = k_begin; k < K; ++k) {
    acc += __half2float(row[k]) * __half2float(x[k]);
  }
  return acc;
}

// Safe vectorized path: __half2 loads only when BOTH row[k] and x[k] are
// 4-byte aligned. Otherwise scalar. No alignment is assumed a priori.
__device__ __forceinline__ float fp16_row_dot_vec2_safe(
    const __half* __restrict__ row,
    const __half* __restrict__ x,
    int K) {
  float acc = 0.0f;
  int k = 0;

  // If both pointers are 2-mod-4, one scalar step aligns both to 0-mod-4.
  if (K >= 1 && !is_half2_aligned(row) && !is_half2_aligned(x)) {
    acc += __half2float(row[0]) * __half2float(x[0]);
    ++k;
  }

  // Vector main loop only when dual alignment holds at this k.
  if (k < K && is_half2_aligned(row + k) && is_half2_aligned(x + k)) {
    for (; k + 1 < K; k += 2) {
      const __half2 w2 = *reinterpret_cast<const __half2*>(row + k);
      const __half2 x2 = *reinterpret_cast<const __half2*>(x + k);
      acc += __low2float(w2) * __low2float(x2) + __high2float(w2) * __high2float(x2);
    }
  }

  // Remainder, or full scalar if vector path never entered.
  acc += fp16_row_dot_scalar(row, x, k, K);
  return acc;
}

__global__ void fp16_gemv_vec2_kernel(
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
  y[n] = fp16_row_dot_vec2_safe(row, x, K);
}

// Explicit host-selected scalar fallback kernel (identical to naive mapping).
__global__ void fp16_gemv_vec2_scalar_fallback_kernel(
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
  y[n] = fp16_row_dot_scalar(row, x, 0, K);
}

bool pointers_allow_any_half2(const __half* W, const __half* x, int N, int K) {
  // Conservative host check: if neither base can ever satisfy dual 4B alignment
  // with the other, force the scalar fallback kernel. Per-row checks still run
  // inside the vector kernel when we do launch it.
  const auto aw = reinterpret_cast<std::uintptr_t>(W) & 0x3u;
  const auto ax = reinterpret_cast<std::uintptr_t>(x) & 0x3u;
  if (K < 2) {
    return false;
  }
  // Bases must be 2B-aligned for __half; if either is weirdly aligned, refuse.
  if ((reinterpret_cast<std::uintptr_t>(W) & 0x1u) != 0u ||
      (reinterpret_cast<std::uintptr_t>(x) & 0x1u) != 0u) {
    return false;
  }
  // If bases differ by 2 mod 4, dual alignment at the same k is impossible.
  if (aw != ax) {
    return false;
  }
  (void)N;
  return true;
}

}  // namespace

void launch_fp16_gemv_vec2(
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

  if (!pointers_allow_any_half2(W, x, N, K)) {
    fp16_gemv_vec2_scalar_fallback_kernel<<<blocks, threads, 0, stream>>>(W, x, y, N, K);
    return;
  }
  fp16_gemv_vec2_kernel<<<blocks, threads, 0, stream>>>(W, x, y, N, K);
}

}  // namespace qkern
