#include "qkern/fp16_gemv.cuh"
#include "qkern/int8_gemv.cuh"

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cstdint>
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

void validate_int8_wq_x(const torch::Tensor& W_q, const torch::Tensor& x) {
  TORCH_CHECK(W_q.is_cuda(), "int8_gemv_fused: W_q must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "int8_gemv_fused: x must be a CUDA tensor");
  TORCH_CHECK(W_q.device() == x.device(), "int8_gemv_fused: W_q and x must be on the same device");
  TORCH_CHECK(W_q.scalar_type() == at::kChar, "int8_gemv_fused: W_q must be int8, got ", W_q.scalar_type());
  TORCH_CHECK(x.scalar_type() == at::kHalf, "int8_gemv_fused: x must be float16, got ", x.scalar_type());
  TORCH_CHECK(W_q.dim() == 2, "int8_gemv_fused: W_q must be [N, K], got shape ", W_q.sizes());
  TORCH_CHECK(x.dim() == 1, "int8_gemv_fused: x must be [K], got shape ", x.sizes());
  TORCH_CHECK(
      W_q.size(1) == x.size(0),
      "int8_gemv_fused: K mismatch: W_q.shape[1]=",
      W_q.size(1),
      " x.shape[0]=",
      x.size(0));
  TORCH_CHECK(W_q.is_contiguous(), "int8_gemv_fused: W_q must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "int8_gemv_fused: x must be contiguous");
  TORCH_CHECK(W_q.size(0) > 0 && W_q.size(1) > 0, "int8_gemv_fused: N and K must be positive");
}

torch::Tensor int8_gemv_fused_cuda(
    torch::Tensor W_q,
    torch::Tensor x,
    torch::Tensor scales,
    const std::string& granularity,
    c10::optional<int64_t> group_size_opt) {
  validate_int8_wq_x(W_q, x);

  TORCH_CHECK(
      scales.scalar_type() == at::kFloat || scales.scalar_type() == at::kHalf ||
          scales.scalar_type() == at::kDouble,
      "int8_gemv_fused: scales must be floating, got ",
      scales.scalar_type());

  const c10::cuda::CUDAGuard device_guard(W_q.device());
  const int N = static_cast<int>(W_q.size(0));
  const int K = static_cast<int>(W_q.size(1));
  auto y = torch::empty({N}, x.options().dtype(at::kFloat));

  const auto* dW = W_q.data_ptr<std::int8_t>();
  const auto* dx = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  float* dy = y.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (granularity == "tensor") {
    TORCH_CHECK(scales.numel() == 1, "int8_gemv_fused: tensor scales must have numel==1, got ", scales.numel());
    const float scale_f = scales.to(at::kFloat).reshape({}).item<float>();
    qkern::launch_int8_gemv_per_tensor_fused(dW, dx, dy, scale_f, N, K, stream);
  } else if (granularity == "channel") {
    auto s = scales.to(at::kFloat).contiguous();
    TORCH_CHECK(
        s.numel() == N && (s.dim() == 1 || (s.dim() == 2 && s.size(1) == 1)),
        "int8_gemv_fused: channel scales must be [N] or [N,1], got ",
        s.sizes());
    if (!s.is_cuda()) {
      s = s.to(W_q.device());
    } else {
      TORCH_CHECK(s.device() == W_q.device(), "int8_gemv_fused: scales device mismatch");
    }
    s = s.reshape({N}).contiguous();
    qkern::launch_int8_gemv_per_channel_fused(dW, dx, dy, s.data_ptr<float>(), N, K, stream);
  } else if (granularity == "group") {
    TORCH_CHECK(group_size_opt.has_value(), "int8_gemv_fused: group_size required for granularity='group'");
    const int group_size = static_cast<int>(group_size_opt.value());
    TORCH_CHECK(group_size > 0, "int8_gemv_fused: group_size must be > 0");
    const int num_groups = (K + group_size - 1) / group_size;
    auto s = scales.to(at::kFloat).contiguous();
    TORCH_CHECK(
        s.dim() == 2 && s.size(0) == N && s.size(1) == num_groups,
        "int8_gemv_fused: group scales must be [N, ceil(K/gs)] = [",
        N,
        ", ",
        num_groups,
        "], got ",
        s.sizes());
    if (!s.is_cuda()) {
      s = s.to(W_q.device());
    } else {
      TORCH_CHECK(s.device() == W_q.device(), "int8_gemv_fused: scales device mismatch");
    }
    s = s.contiguous();
    qkern::launch_int8_gemv_per_group_fused(
        dW, dx, dy, s.data_ptr<float>(), N, K, group_size, stream);
  } else {
    TORCH_CHECK(
        false,
        "int8_gemv_fused: unknown granularity '",
        granularity,
        "' (expected 'tensor', 'channel', or 'group')");
  }

  C10_CUDA_CHECK(cudaGetLastError());
  return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "QKern CUDA extension (FP16 GEMV variants + INT8 fused GEMV)";
  m.def(
      "fp16_gemv",
      &fp16_gemv_cuda,
      "FP16 GEMV with FP32 accumulation (CUDA)",
      pybind11::arg("W"),
      pybind11::arg("x"),
      pybind11::arg("variant") = "naive");
  m.def(
      "int8_gemv_fused",
      &int8_gemv_fused_cuda,
      "Fused INT8 weight-only GEMV (tensor / channel / group)",
      pybind11::arg("W_q"),
      pybind11::arg("x"),
      pybind11::arg("scales"),
      pybind11::arg("granularity") = "tensor",
      pybind11::arg("group_size") = pybind11::none());
}
