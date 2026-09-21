#include "qkern/int4_gemv.cuh"

namespace qkern {
namespace {

// ---------------------------------------------------------------------------
// Device helpers (kept small and separate so later opts can swap load paths
// without rewriting the accumulation math).
// ---------------------------------------------------------------------------

// Sign-extend a 4-bit two's-complement nibble to signed int in [-8, 7].
__device__ __forceinline__ int sign_extend_nibble(unsigned int nibble) {
  const int v = static_cast<int>(nibble & 0xFu);
  return (v >= 8) ? (v - 16) : v;
}

// Unpack one packed byte into two signed INT4 values (Phase-1 layout).
// Future: vectorized multi-byte loads can feed this helper.
__device__ __forceinline__ void unpack_int4_byte(
    unsigned int packed_byte,
    int* q_lo,
    int* q_hi) {
  *q_lo = sign_extend_nibble(packed_byte);
  *q_hi = sign_extend_nibble(packed_byte >> 4);
}

// Group index for logical K. Incomplete final groups are fine (ceil in layout).
// Future: replace divide with shift when group_size is a power of two.
__device__ __forceinline__ int group_index(int k, int group_size) {
  return k / group_size;
}

constexpr int kThreadsPerBlock = 256;

// ---------------------------------------------------------------------------
// Kernels — correctness-first: one thread per output row.
// ---------------------------------------------------------------------------

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

  // Future: stage x (and optionally scales) in shared memory for reuse.
  const std::uint8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k);

  float acc = 0.0f;
  for (int i = 0; i < packed_k; ++i) {
    // Future: vectorized weight loads (e.g. uint32 / uint4) into registers.
    const unsigned int p = static_cast<unsigned int>(row[i]);
    int q_lo = 0;
    int q_hi = 0;
    unpack_int4_byte(p, &q_lo, &q_hi);

    const int k0 = i * 2;
    // Future: vectorized activation loads (__half2) when K alignment allows.
    acc += static_cast<float>(q_lo) * __half2float(x[k0]);
    if (k0 + 1 < K) {
      acc += static_cast<float>(q_hi) * __half2float(x[k0 + 1]);
    }
  }
  y[n] = acc * scale;
}

__global__ void int4_gemv_per_channel_fused_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scales,
    int N,
    int K,
    int packed_k) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= N) {
    return;
  }

  const std::uint8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k);
  // One scale per output row — cheap; keep simple for now.
  const float scale = scales[n];

  float acc = 0.0f;
  for (int i = 0; i < packed_k; ++i) {
    const unsigned int p = static_cast<unsigned int>(row[i]);
    int q_lo = 0;
    int q_hi = 0;
    unpack_int4_byte(p, &q_lo, &q_hi);

    const int k0 = i * 2;
    acc += static_cast<float>(q_lo) * __half2float(x[k0]);
    if (k0 + 1 < K) {
      acc += static_cast<float>(q_hi) * __half2float(x[k0 + 1]);
    }
  }
  y[n] = acc * scale;
}

__global__ void int4_gemv_per_group_fused_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scales,
    int N,
    int K,
    int packed_k,
    int group_size,
    int num_groups) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= N) {
    return;
  }

  const std::uint8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k);
  // Future: cache row scales in registers / shared memory for the active groups.
  const float* row_scales =
      scales + static_cast<std::size_t>(n) * static_cast<std::size_t>(num_groups);

  float acc = 0.0f;
  for (int i = 0; i < packed_k; ++i) {
    const unsigned int p = static_cast<unsigned int>(row[i]);
    int q_lo = 0;
    int q_hi = 0;
    unpack_int4_byte(p, &q_lo, &q_hi);

    const int k0 = i * 2;
    // Per-weight group scale (correctness-first; not coalesced/staged yet).
    {
      const int g0 = group_index(k0, group_size);
      const float s0 = row_scales[g0];
      acc += (static_cast<float>(q_lo) * s0) * __half2float(x[k0]);
    }
    if (k0 + 1 < K) {
      const int g1 = group_index(k0 + 1, group_size);
      const float s1 = row_scales[g1];
      acc += (static_cast<float>(q_hi) * s1) * __half2float(x[k0 + 1]);
    }
  }
  y[n] = acc;
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
  const int blocks = (N + kThreadsPerBlock - 1) / kThreadsPerBlock;
  int4_gemv_per_tensor_fused_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      W_q_packed, x, y, scale, N, K, packed_k);
}

void launch_int4_gemv_per_channel_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  const int packed_k = (K + 1) / 2;
  const int blocks = (N + kThreadsPerBlock - 1) / kThreadsPerBlock;
  int4_gemv_per_channel_fused_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      W_q_packed, x, y, scales, N, K, packed_k);
}

void launch_int4_gemv_per_group_fused(
    const std::uint8_t* W_q_packed,
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
  const int packed_k = (K + 1) / 2;
  const int num_groups = (K + group_size - 1) / group_size;
  const int blocks = (N + kThreadsPerBlock - 1) / kThreadsPerBlock;
  int4_gemv_per_group_fused_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      W_q_packed, x, y, scales, N, K, packed_k, group_size, num_groups);
}

}  // namespace qkern
