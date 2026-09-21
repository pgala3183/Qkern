#include "qkern/fp16_gemv.cuh"

#include <stdexcept>

namespace qkern {

void launch_fp16_gemv(
    Fp16GemvVariant variant,
    const __half* W,
    const __half* x,
    float* y,
    int N,
    int K,
    cudaStream_t stream) {
  switch (variant) {
    case Fp16GemvVariant::Naive:
      launch_fp16_gemv_naive(W, x, y, N, K, stream);
      return;
    case Fp16GemvVariant::XSmem:
      launch_fp16_gemv_x_smem(W, x, y, N, K, stream);
      return;
    default:
      throw std::invalid_argument("launch_fp16_gemv: unknown variant");
  }
}

}  // namespace qkern
