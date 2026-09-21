#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qkern {

// FP16 GEMV variants. Keep naive as the permanent Phase-2 baseline.
enum class Fp16GemvVariant {
  Naive = 0,
  XSmem = 1,  // activation (x) cached in shared memory — first optimization
};

inline const char* fp16_gemv_variant_name(Fp16GemvVariant v) {
  switch (v) {
    case Fp16GemvVariant::Naive:
      return "naive";
    case Fp16GemvVariant::XSmem:
      return "x_smem";
    default:
      return "unknown";
  }
}

// y[N] = W[N,K] * x[K], FP16 inputs, FP32 accumulation.
void launch_fp16_gemv_naive(
    const __half* W,
    const __half* x,
    float* y,
    int N,
    int K,
    cudaStream_t stream = nullptr);

// Same math; stages tiles of x in shared memory to cut redundant global x loads.
void launch_fp16_gemv_x_smem(
    const __half* W,
    const __half* x,
    float* y,
    int N,
    int K,
    cudaStream_t stream = nullptr);

void launch_fp16_gemv(
    Fp16GemvVariant variant,
    const __half* W,
    const __half* x,
    float* y,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace qkern
