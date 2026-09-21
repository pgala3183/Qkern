#include "qkern/int8_gemv.cuh"

namespace qkern {
namespace {

// Correctness-first fused INT8 GEMV: one thread per output row.
// Scale access is intentionally simple (no shared-memory staging of scales yet).

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
    acc += static_cast<float>(row[k]) * __half2float(x[k]);
  }
  y[n] = acc * scale;
}

__global__ void int8_gemv_per_channel_fused_kernel(
    const std::int8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scales,
    int N,
    int K) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= N) {
    return;
  }

  const std::int8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(K);
  const float scale = scales[n];
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += static_cast<float>(row[k]) * __half2float(x[k]);
  }
  y[n] = acc * scale;
}

__global__ void int8_gemv_per_group_fused_kernel(
    const std::int8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scales,
    int N,
    int K,
    int group_size,
    int num_groups) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= N) {
    return;
  }

  const std::int8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(K);
  const float* row_scales =
      scales + static_cast<std::size_t>(n) * static_cast<std::size_t>(num_groups);

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const int g = k / group_size;  // last group may be shorter when K % gs != 0
    const float scale = row_scales[g];
    acc += (static_cast<float>(row[k]) * scale) * __half2float(x[k]);
  }
  y[n] = acc;
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

void launch_int8_gemv_per_channel_fused(
    const std::int8_t* W_q,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  int8_gemv_per_channel_fused_kernel<<<blocks, threads, 0, stream>>>(
      W_q, x, y, scales, N, K);
}

void launch_int8_gemv_per_group_fused(
    const std::int8_t* W_q,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    int group_size,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0 || group_size <= 0) {
    return;
  }
  const int num_groups = (K + group_size - 1) / group_size;
  constexpr int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  int8_gemv_per_group_fused_kernel<<<blocks, threads, 0, stream>>>(
      W_q, x, y, scales, N, K, group_size, num_groups);
}

}  // namespace qkern
