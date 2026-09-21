#include "qkern/int4_gemv.cuh"

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

__device__ __forceinline__ int group_index(int k, int group_size) {
  return k / group_size;
}

template <int BlockN, int BlockK>
__device__ void load_x_tile_to_smem(
    const __half* __restrict__ x,
    __half* __restrict__ smem_stage,
    int k_base,
    int K) {
  for (int t = static_cast<int>(threadIdx.x); t < BlockK; t += BlockN) {
    const int k = k_base + t;
    smem_stage[t] = (k < K) ? x[k] : __float2half(0.0f);
  }
}

template <int VecSize, typename ScaleFn>
__device__ float accumulate_row_range(
    const std::uint8_t* __restrict__ row,
    const __half* __restrict__ x_base,
    int k_begin,
    int k_end,
    int k_offset,
    int packed_k,
    ScaleFn&& scale_at_k) {
  float acc = 0.0f;
  int i = k_begin / 2;
  const int i_end = (k_end + 1) / 2;

  while (i + VecSize <= i_end && i + VecSize <= packed_k) {
#pragma unroll
    for (int v = 0; v < VecSize; ++v) {
      const unsigned int p = static_cast<unsigned int>(row[i + v]);
      int q_lo = 0;
      int q_hi = 0;
      unpack_int4_byte(p, &q_lo, &q_hi);
      const int k0 = (i + v) * 2;
      if (k0 >= k_begin && k0 < k_end) {
        const float s = scale_at_k(k0);
        acc += (static_cast<float>(q_lo) * s) * __half2float(x_base[k0 - k_offset]);
      }
      if (k0 + 1 >= k_begin && k0 + 1 < k_end) {
        const float s = scale_at_k(k0 + 1);
        acc += (static_cast<float>(q_hi) * s) * __half2float(x_base[k0 + 1 - k_offset]);
      }
    }
    i += VecSize;
  }
  for (; i < i_end && i < packed_k; ++i) {
    const unsigned int p = static_cast<unsigned int>(row[i]);
    int q_lo = 0;
    int q_hi = 0;
    unpack_int4_byte(p, &q_lo, &q_hi);
    const int k0 = i * 2;
    if (k0 >= k_begin && k0 < k_end) {
      const float s = scale_at_k(k0);
      acc += (static_cast<float>(q_lo) * s) * __half2float(x_base[k0 - k_offset]);
    }
    if (k0 + 1 >= k_begin && k0 + 1 < k_end) {
      const float s = scale_at_k(k0 + 1);
      acc += (static_cast<float>(q_hi) * s) * __half2float(x_base[k0 + 1 - k_offset]);
    }
  }
  return acc;
}

// =============================================================================
// No-smem kernels (BLOCK_K == 0) — default config uses these, matching the
// original scalar packed-load implementation.
// =============================================================================

template <int BlockN>
__global__ void int4_gemv_tensor_nosmem_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    float scale,
    int N,
    int K,
    int packed_k) {
  const int n = static_cast<int>(blockIdx.x * BlockN + threadIdx.x);
  if (n >= N) {
    return;
  }
  const std::uint8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k);
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

template <int BlockN>
__global__ void int4_gemv_channel_nosmem_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scales,
    int N,
    int K,
    int packed_k) {
  const int n = static_cast<int>(blockIdx.x * BlockN + threadIdx.x);
  if (n >= N) {
    return;
  }
  const std::uint8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k);
  const float row_scale = scales[n];
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
  y[n] = acc * row_scale;
}

template <int BlockN>
__global__ void int4_gemv_group_nosmem_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scales,
    int N,
    int K,
    int packed_k,
    int group_size,
    int num_groups) {
  const int n = static_cast<int>(blockIdx.x * BlockN + threadIdx.x);
  if (n >= N) {
    return;
  }
  const std::uint8_t* row =
      W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k);
  const float* row_scales =
      scales + static_cast<std::size_t>(n) * static_cast<std::size_t>(num_groups);
  float acc = 0.0f;
  for (int i = 0; i < packed_k; ++i) {
    const unsigned int p = static_cast<unsigned int>(row[i]);
    int q_lo = 0;
    int q_hi = 0;
    unpack_int4_byte(p, &q_lo, &q_hi);
    const int k0 = i * 2;
    {
      const float s0 = row_scales[group_index(k0, group_size)];
      acc += (static_cast<float>(q_lo) * s0) * __half2float(x[k0]);
    }
    if (k0 + 1 < K) {
      const float s1 = row_scales[group_index(k0 + 1, group_size)];
      acc += (static_cast<float>(q_hi) * s1) * __half2float(x[k0 + 1]);
    }
  }
  y[n] = acc;
}

// =============================================================================
// Smem-tiled kernels (BLOCK_K > 0)
// =============================================================================

template <typename Config>
__global__ void int4_gemv_tensor_smem_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    float scale,
    int N,
    int K,
    int packed_k) {
  constexpr int BlockN = Config::BLOCK_N;
  constexpr int BlockK = Config::BLOCK_K;
  constexpr int VecSize = Config::VEC_SIZE;
  constexpr int NumStages = Config::NUM_STAGES;
  static_assert(BlockK > 0, "smem kernel requires BLOCK_K > 0");

  extern __shared__ __half smem_x[];

  const int n = static_cast<int>(blockIdx.x * BlockN + threadIdx.x);
  const bool active = (n < N);
  const std::uint8_t* row =
      active ? (W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k))
             : nullptr;
  auto scale_fn = [](int /*k*/) -> float { return 1.0f; };

  float acc = 0.0f;
  for (int k_base = 0; k_base < K; k_base += BlockK) {
    const int stage = (k_base / BlockK) % NumStages;
    __half* stage_ptr = smem_x + stage * BlockK;
    load_x_tile_to_smem<BlockN, BlockK>(x, stage_ptr, k_base, K);
    __syncthreads();
    if (active) {
      const int k_end = (k_base + BlockK < K) ? (k_base + BlockK) : K;
      acc += accumulate_row_range<VecSize>(
          row, stage_ptr, k_base, k_end, k_base, packed_k, scale_fn);
    }
    __syncthreads();
  }
  if (active) {
    y[n] = acc * scale;
  }
}

template <typename Config>
__global__ void int4_gemv_channel_smem_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scales,
    int N,
    int K,
    int packed_k) {
  constexpr int BlockN = Config::BLOCK_N;
  constexpr int BlockK = Config::BLOCK_K;
  constexpr int VecSize = Config::VEC_SIZE;
  constexpr int NumStages = Config::NUM_STAGES;
  static_assert(BlockK > 0, "smem kernel requires BLOCK_K > 0");

  extern __shared__ __half smem_x[];

  const int n = static_cast<int>(blockIdx.x * BlockN + threadIdx.x);
  const bool active = (n < N);
  const std::uint8_t* row =
      active ? (W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k))
             : nullptr;
  const float row_scale = active ? scales[n] : 0.0f;
  auto scale_fn = [](int /*k*/) -> float { return 1.0f; };

  float acc = 0.0f;
  for (int k_base = 0; k_base < K; k_base += BlockK) {
    const int stage = (k_base / BlockK) % NumStages;
    __half* stage_ptr = smem_x + stage * BlockK;
    load_x_tile_to_smem<BlockN, BlockK>(x, stage_ptr, k_base, K);
    __syncthreads();
    if (active) {
      const int k_end = (k_base + BlockK < K) ? (k_base + BlockK) : K;
      acc += accumulate_row_range<VecSize>(
          row, stage_ptr, k_base, k_end, k_base, packed_k, scale_fn);
    }
    __syncthreads();
  }
  if (active) {
    y[n] = acc * row_scale;
  }
}

template <typename Config>
__global__ void int4_gemv_group_smem_kernel(
    const std::uint8_t* __restrict__ W_q,
    const __half* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scales,
    int N,
    int K,
    int packed_k,
    int group_size,
    int num_groups) {
  constexpr int BlockN = Config::BLOCK_N;
  constexpr int BlockK = Config::BLOCK_K;
  constexpr int VecSize = Config::VEC_SIZE;
  constexpr int NumStages = Config::NUM_STAGES;
  static_assert(BlockK > 0, "smem kernel requires BLOCK_K > 0");

  extern __shared__ __half smem_x[];

  const int n = static_cast<int>(blockIdx.x * BlockN + threadIdx.x);
  const bool active = (n < N);
  const std::uint8_t* row =
      active ? (W_q + static_cast<std::size_t>(n) * static_cast<std::size_t>(packed_k))
             : nullptr;
  const float* row_scales =
      active ? (scales + static_cast<std::size_t>(n) * static_cast<std::size_t>(num_groups))
             : nullptr;
  auto scale_fn = [&](int k) -> float {
    return row_scales[group_index(k, group_size)];
  };

  float acc = 0.0f;
  for (int k_base = 0; k_base < K; k_base += BlockK) {
    const int stage = (k_base / BlockK) % NumStages;
    __half* stage_ptr = smem_x + stage * BlockK;
    load_x_tile_to_smem<BlockN, BlockK>(x, stage_ptr, k_base, K);
    __syncthreads();
    if (active) {
      const int k_end = (k_base + BlockK < K) ? (k_base + BlockK) : K;
      acc += accumulate_row_range<VecSize>(
          row, stage_ptr, k_base, k_end, k_base, packed_k, scale_fn);
    }
    __syncthreads();
  }
  if (active) {
    y[n] = acc;
  }
}

template <typename Config>
std::size_t smem_bytes_for_config() {
  if constexpr (Config::BLOCK_K == 0) {
    return 0;
  } else {
    return static_cast<std::size_t>(Config::NUM_STAGES) *
           static_cast<std::size_t>(Config::BLOCK_K) * sizeof(__half);
  }
}

template <typename Config>
void launch_tensor(
    const std::uint8_t* W_q,
    const __half* x,
    float* y,
    float scale,
    int N,
    int K,
    cudaStream_t stream) {
  const int packed_k = (K + 1) / 2;
  const int blocks = (N + Config::BLOCK_N - 1) / Config::BLOCK_N;
  if constexpr (Config::BLOCK_K == 0) {
    int4_gemv_tensor_nosmem_kernel<Config::BLOCK_N>
        <<<blocks, Config::BLOCK_N, 0, stream>>>(W_q, x, y, scale, N, K, packed_k);
  } else {
    const std::size_t smem = smem_bytes_for_config<Config>();
    int4_gemv_tensor_smem_kernel<Config>
        <<<blocks, Config::BLOCK_N, smem, stream>>>(W_q, x, y, scale, N, K, packed_k);
  }
}

template <typename Config>
void launch_channel(
    const std::uint8_t* W_q,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    cudaStream_t stream) {
  const int packed_k = (K + 1) / 2;
  const int blocks = (N + Config::BLOCK_N - 1) / Config::BLOCK_N;
  if constexpr (Config::BLOCK_K == 0) {
    int4_gemv_channel_nosmem_kernel<Config::BLOCK_N>
        <<<blocks, Config::BLOCK_N, 0, stream>>>(W_q, x, y, scales, N, K, packed_k);
  } else {
    const std::size_t smem = smem_bytes_for_config<Config>();
    int4_gemv_channel_smem_kernel<Config>
        <<<blocks, Config::BLOCK_N, smem, stream>>>(W_q, x, y, scales, N, K, packed_k);
  }
}

template <typename Config>
void launch_group(
    const std::uint8_t* W_q,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    int group_size,
    cudaStream_t stream) {
  const int packed_k = (K + 1) / 2;
  const int num_groups = (K + group_size - 1) / group_size;
  const int blocks = (N + Config::BLOCK_N - 1) / Config::BLOCK_N;
  if constexpr (Config::BLOCK_K == 0) {
    int4_gemv_group_nosmem_kernel<Config::BLOCK_N><<<blocks, Config::BLOCK_N, 0, stream>>>(
        W_q, x, y, scales, N, K, packed_k, group_size, num_groups);
  } else {
    const std::size_t smem = smem_bytes_for_config<Config>();
    int4_gemv_group_smem_kernel<Config><<<blocks, Config::BLOCK_N, smem, stream>>>(
        W_q, x, y, scales, N, K, packed_k, group_size, num_groups);
  }
}

}  // namespace

void launch_int4_gemv_per_tensor_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    float scale,
    int N,
    int K,
    Int4GemvConfigId config,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  switch (config) {
    case Int4GemvConfigId::Bn128Bk128V1S1:
      launch_tensor<Int4GemvConfigBn128Bk128V1S1>(W_q_packed, x, y, scale, N, K, stream);
      break;
    case Int4GemvConfigId::Bn256Bk256V4S2:
      launch_tensor<Int4GemvConfigBn256Bk256V4S2>(W_q_packed, x, y, scale, N, K, stream);
      break;
    case Int4GemvConfigId::Default:
    default:
      launch_tensor<Int4GemvConfigDefault>(W_q_packed, x, y, scale, N, K, stream);
      break;
  }
}

void launch_int4_gemv_per_channel_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    Int4GemvConfigId config,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0) {
    return;
  }
  switch (config) {
    case Int4GemvConfigId::Bn128Bk128V1S1:
      launch_channel<Int4GemvConfigBn128Bk128V1S1>(W_q_packed, x, y, scales, N, K, stream);
      break;
    case Int4GemvConfigId::Bn256Bk256V4S2:
      launch_channel<Int4GemvConfigBn256Bk256V4S2>(W_q_packed, x, y, scales, N, K, stream);
      break;
    case Int4GemvConfigId::Default:
    default:
      launch_channel<Int4GemvConfigDefault>(W_q_packed, x, y, scales, N, K, stream);
      break;
  }
}

void launch_int4_gemv_per_group_fused(
    const std::uint8_t* W_q_packed,
    const __half* x,
    float* y,
    const float* scales,
    int N,
    int K,
    int group_size,
    Int4GemvConfigId config,
    cudaStream_t stream) {
  if (N <= 0 || K <= 0 || group_size <= 0) {
    return;
  }
  switch (config) {
    case Int4GemvConfigId::Bn128Bk128V1S1:
      launch_group<Int4GemvConfigBn128Bk128V1S1>(
          W_q_packed, x, y, scales, N, K, group_size, stream);
      break;
    case Int4GemvConfigId::Bn256Bk256V4S2:
      launch_group<Int4GemvConfigBn256Bk256V4S2>(
          W_q_packed, x, y, scales, N, K, group_size, stream);
      break;
    case Int4GemvConfigId::Default:
    default:
      launch_group<Int4GemvConfigDefault>(
          W_q_packed, x, y, scales, N, K, group_size, stream);
      break;
  }
}

}  // namespace qkern
