// Phase 0 smoke: confirm NVRTC headers/libs work for later runtime compilation.

#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die_nvrtc(const char* what, nvrtcResult result) {
  std::fprintf(stderr, "qkern_nvrtc_smoke: %s: %s\n", what, nvrtcGetErrorString(result));
  std::exit(1);
}

}  // namespace

int main() {
  int nvrtc_major = 0;
  int nvrtc_minor = 0;
  nvrtcResult result = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
  if (result != NVRTC_SUCCESS) {
    die_nvrtc("nvrtcVersion", result);
  }

  const char* src =
      "extern \"C\" __global__ void qkern_add_one(int* x) {\n"
      "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
      "    x[0] += 1;\n"
      "  }\n"
      "}\n";

  nvrtcProgram prog = nullptr;
  result = nvrtcCreateProgram(&prog, src, "qkern_add_one.cu", 0, nullptr, nullptr);
  if (result != NVRTC_SUCCESS) {
    die_nvrtc("nvrtcCreateProgram", result);
  }

  // Compile for the installed GPU's architecture when possible.
  int device = 0;
  cudaError_t cerr = cudaGetDevice(&device);
  if (cerr != cudaSuccess) {
    std::fprintf(stderr, "qkern_nvrtc_smoke: cudaGetDevice: %s\n", cudaGetErrorString(cerr));
    nvrtcDestroyProgram(&prog);
    return 1;
  }

  cudaDeviceProp prop{};
  cerr = cudaGetDeviceProperties(&prop, device);
  if (cerr != cudaSuccess) {
    std::fprintf(stderr, "qkern_nvrtc_smoke: cudaGetDeviceProperties: %s\n", cudaGetErrorString(cerr));
    nvrtcDestroyProgram(&prog);
    return 1;
  }

  const std::string arch_flag =
      std::string("--gpu-architecture=sm_") + std::to_string(prop.major) + std::to_string(prop.minor);
  const char* opts[] = {arch_flag.c_str()};

  result = nvrtcCompileProgram(prog, 1, opts);
  if (result != NVRTC_SUCCESS) {
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::vector<char> log(log_size);
    nvrtcGetProgramLog(prog, log.data());
    std::fprintf(stderr, "qkern_nvrtc_smoke: compile failed:\n%s\n", log.data());
    nvrtcDestroyProgram(&prog);
    die_nvrtc("nvrtcCompileProgram", result);
  }

  size_t ptx_size = 0;
  result = nvrtcGetPTXSize(prog, &ptx_size);
  if (result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    die_nvrtc("nvrtcGetPTXSize", result);
  }

  std::vector<char> ptx(ptx_size);
  result = nvrtcGetPTX(prog, ptx.data());
  if (result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    die_nvrtc("nvrtcGetPTX", result);
  }

  nvrtcDestroyProgram(&prog);

  std::printf("QKern NVRTC smoke (Phase 0)\n");
  std::printf("NVRTC version: %d.%d\n", nvrtc_major, nvrtc_minor);
  std::printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("Compiled arch flag: %s\n", arch_flag.c_str());
  std::printf("PTX bytes: %zu\n", ptx_size);
  std::printf("NVRTC OK\n");
  return 0;
}
