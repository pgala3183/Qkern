#include "qkern/int4_dequant.cuh"

namespace qkern {
namespace {

__device__ __forceinline__ int sign_extend_nibble(unsigned int nibble) {
  const int v = static_cast<int>(nibble & 0xFu);
  return (v >= 8) ? (v - 16) : v;
}

__device__ __forceinline__ void unpack_int4_byte(
    unsigned int packed_byte,
    int* q_lo,
    int* q_hi) {
  *q_lo = sign_extend_nibble(packed_byte);
  *q_hi = sign_extend_nibble(packed_byte >> 4);
}

// One thread per packed byte → writes up to two FP16 outputs.
// Correctness-first; not aggressively optimized.

__global__ void int4_dequant_per_tensor_kernel(
    const std::uint8_t* __restrict__ W_q,
    __half* __restrict__ W_hat,
    float scale,
    int N,
    int K,
    int packed_k) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = N * packed_k;
  if (idx >= total) {
    return;
  }
  const int n = idx / packed_k;
  const int i = idx - n * packed_k;
  const unsigned int p = static_cast<unsigned int>(W_q[idx]);
  int q_lo = 0;
  int q_hi = 0;
  unpack_int4_byte(p, &q_lo, &q_hi);

  const int k0 = i * 2;
  __half* out = W_hat + static_cast<std::size_t>(n) * static_cast<std::size_t>(K);
  out[k0] = __float2half(static_cast<float>(q_lo) * scale);
  if (k0 + 1 < K) {
    out[k0 + 1] = __float2half(static_cast<float>(q_hi) * scale);
  }
}

__global__ void int4_dequant_per_channel_kernel(
    const std::uint8_t* __restrict__ W_q,
    __half* __restrict__ W_hat,
    const float* __restrict__ scales,
    int N,
    int K,
    int packed_k) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = N * packed_k;
  if (idx >= total) {
    return;
  }
  const int n = idx / packed_k;
  const int i = idx - n * packed_k;
  const float scale = scales[n];
  const unsigned int p = static_cast<unsigned int>(W_q[idx]);
  int q_lo = 0;
  int q_hi = 0;
  unpack_int4_byte(p, &q_lo, &q_hi);

  const int k0 = i * 2;
  __half* out = W_hat + static_cast<std::size_t>(n) * static_cast<std::size_t>(K);
  out[k0] = __float2half(static_cast<float>(q_lo) * scale);
  if (k0 + 1 < K) {
    out[k0 + 1] = __float2half(static_cast<float>(q_hi) * scale);
  }
}

__global__ void int4_dequant_per_group_kernel(
    const std::uint8_t* __restrict__ W_q,
    __half* __restrict__ W_hat,
    const float* __restrict__ scales,
    int N,
    int K,
    int packed_k,
    int group_size,
    int num_groups) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = N * packed_k;
  if (idx >= total) {
    return;
  }
  const int n = idx / packed_k;
  const int i = idx - n * packed_k;
  const float* row_scales =
      scales + static_cast<std::size_t>(n) * static_cast<std::size_t>(num_groups);

  const unsigned int p = static_cast<unsigned int>(W_q[idx]);
  int q_lo = 0;
  int q_hi = 0;
  unpack_int4_byte(p, &q_lo, &q_hi);

  const int k0 = i * 2;
  __half* out = W_hat + static_cast<std::size_t>(n) * static_cast<std::size_t>(K);
  {
    const float s0 = row_scales[k0 / group_size];
    out[k0] = __float2half(static_cast<float>(q_lo) * s0);
  }
  if (k0 + 1 < K) {
    const float s1 = row_scales[(k0 + 1) / group_size];
    out[k0 + 1] = __float2half(static_cast<float>(q_hi) * s1);
  }
}

constexpr int kThreads = 256;

}  // namespace

void launch_int4_dequant_per_tensor(
    const std::uint8_t* W_q_packed,
    __half* W_hat,
    float scale,
    int N,
    int K,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  const int packed_k = (K + 1) / 2;
  const int total = N * packed_k;
  const int blocks = (total + kThreads - 1) / kThreads;
  int4_dequant_per_tensor_kernel<<<blocks, kThreads, 0, stream>>>(
      W_q_packed, W_hat, scale, N, K, packed_k);
}

void launch_int4_dequant_per_channel(
    const std::uint8_t* W_q_packed,
    __half* W_hat,
    const float* scales,
    int N,
    int K,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  const int packed_k = (K + 1) / 2;
  const int total = N * packed_k;
  const int blocks = (total + kThreads - 1) / kThreads;
  int4_dequant_per_channel_kernel<<<blocks, kThreads, 0, stream>>>(
      W_q_packed, W_hat, scales, N, K, packed_k);
}

void launch_int4_dequant_per_group(
    const std::uint8_t* W_q_packed,
    __half* W_hat,
    const float* scales,
    int N,
    int K,
    int group_size,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0 || group_size <= 0) {
    return;
  }
  const int packed_k = (K + 1) / 2;
  const int num_groups = (K + group_size - 1) / group_size;
  const int total = N * packed_k;
  const int blocks = (total + kThreads - 1) / kThreads;
  int4_dequant_per_group_kernel<<<blocks, kThreads, 0, stream>>>(
      W_q_packed, W_hat, scales, N, K, packed_k, group_size, num_groups);
}

}  // namespace qkern
