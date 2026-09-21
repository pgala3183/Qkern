#include "qkern/fp16_gemv.cuh"

#include <algorithm>

namespace qkern {
namespace {

// First optimization only: cache tiles of x in shared memory.
// Thread mapping matches naive (one thread per output row) so the sole
// algorithmic change is where x is read from after staging.
__global__ void fp16_gemv_x_smem_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    float* __restrict__ y,
    int N,
    int K,
    int tile_k) {
  extern __shared__ __half x_tile[];

  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += tile_k) {
    const int tile = (tile_k < (K - k0)) ? tile_k : (K - k0);

    // All threads in the block participate in staging (including n >= N).
    for (int i = static_cast<int>(threadIdx.x); i < tile; i += static_cast<int>(blockDim.x)) {
      x_tile[i] = x[k0 + i];
    }
    __syncthreads();

    if (n < N) {
      const __half* row =
          W + static_cast<std::size_t>(n) * static_cast<std::size_t>(K) + static_cast<std::size_t>(k0);
      for (int i = 0; i < tile; ++i) {
        const float wv = __half2float(row[i]);
        const float xv = __half2float(x_tile[i]);
        acc += wv * xv;
      }
    }
    __syncthreads();
  }

  if (n < N) {
    y[n] = acc;
  }
}

int choose_tile_k(int K) {
  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  // Leave headroom; do not hardcode architecture-specific caps beyond the query.
  const int max_by_smem = static_cast<int>(prop.sharedMemPerBlock / sizeof(__half));
  int tile = std::min(K, max_by_smem);
  // Prefer a multiple of the block size used below for clean cooperative loads.
  constexpr int threads = 256;
  if (tile > threads) {
    tile = (tile / threads) * threads;
  }
  if (tile < 1) {
    tile = 1;
  }
  return tile;
}

}  // namespace

void launch_fp16_gemv_x_smem(
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
  const int tile_k = choose_tile_k(K);
  const std::size_t smem_bytes = static_cast<std::size_t>(tile_k) * sizeof(__half);
  fp16_gemv_x_smem_kernel<<<blocks, threads, smem_bytes, stream>>>(W, x, y, N, K, tile_k);
}

}  // namespace qkern
