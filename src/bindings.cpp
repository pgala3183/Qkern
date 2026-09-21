#include "qkern/fp16_gemv.cuh"

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <string>

namespace {

void validate_fp16_gemv_inputs(const torch::Tensor& W, const torch::Tensor& x) {
  TORCH_CHECK(W.is_cuda(), "fp16_gemv: W must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "fp16_gemv: x must be a CUDA tensor");
  TORCH_CHECK(W.device() == x.device(), "fp16_gemv: W and x must be on the same device");
  TORCH_CHECK(W.scalar_type() == at::kHalf, "fp16_gemv: W must be float16, got ", W.scalar_type());
  TORCH_CHECK(x.scalar_type() == at::kHalf, "fp16_gemv: x must be float16, got ", x.scalar_type());
  TORCH_CHECK(W.dim() == 2, "fp16_gemv: W must be rank-2 [N, K], got shape ", W.sizes());
  TORCH_CHECK(x.dim() == 1, "fp16_gemv: x must be rank-1 [K], got shape ", x.sizes());
  TORCH_CHECK(W.size(1) == x.size(0), "fp16_gemv: K mismatch: W.shape[1]=", W.size(1), " x.shape[0]=", x.size(0));
  TORCH_CHECK(W.is_contiguous(), "fp16_gemv: W must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "fp16_gemv: x must be contiguous");
  TORCH_CHECK(W.size(0) > 0 && W.size(1) > 0, "fp16_gemv: N and K must be positive");
}

qkern::Fp16GemvVariant parse_variant(const std::string& name) {
  if (name == "naive") {
    return qkern::Fp16GemvVariant::Naive;
  }
  if (name == "x_smem") {
    return qkern::Fp16GemvVariant::XSmem;
  }
  if (name == "vec2") {
    return qkern::Fp16GemvVariant::Vec2;
  }
  TORCH_CHECK(
      false,
      "fp16_gemv: unknown variant '",
      name,
      "' (expected 'naive', 'x_smem', or 'vec2')");
}

torch::Tensor fp16_gemv_cuda(torch::Tensor W, torch::Tensor x, const std::string& variant) {
  validate_fp16_gemv_inputs(W, x);

  const c10::cuda::CUDAGuard device_guard(W.device());
  const int N = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));

  auto y = torch::empty({N}, W.options().dtype(at::kFloat));

  const auto* dW = reinterpret_cast<const __half*>(W.data_ptr<at::Half>());
  const auto* dx = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  float* dy = y.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  qkern::launch_fp16_gemv(parse_variant(variant), dW, dx, dy, N, K, stream);

  C10_CUDA_CHECK(cudaGetLastError());
  return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "QKern CUDA extension (FP16 GEMV variants)";
  m.def(
      "fp16_gemv",
      &fp16_gemv_cuda,
      "FP16 GEMV with FP32 accumulation (CUDA)",
      pybind11::arg("W"),
      pybind11::arg("x"),
      pybind11::arg("variant") = "naive");
}
