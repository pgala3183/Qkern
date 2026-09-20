// Phase 0 smoke: query actual GPU properties at runtime.
// Do not hardcode hardware limits for later autotuner decisions.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void die(const char* what, cudaError_t err) {
  std::fprintf(stderr, "qkern_device_query: %s: %s\n", what, cudaGetErrorString(err));
  std::exit(1);
}

}  // namespace

int main() {
  int driver_version = 0;
  int runtime_version = 0;
  cudaError_t err = cudaDriverGetVersion(&driver_version);
  if (err != cudaSuccess) {
    die("cudaDriverGetVersion", err);
  }
  err = cudaRuntimeGetVersion(&runtime_version);
  if (err != cudaSuccess) {
    die("cudaRuntimeGetVersion", err);
  }

  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    die("cudaGetDeviceCount", err);
  }
  if (device_count < 1) {
    std::fprintf(stderr, "qkern_device_query: no CUDA devices found\n");
    return 1;
  }

  std::printf("QKern device query (Phase 0)\n");
  std::printf("CUDA driver version:  %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
  std::printf("CUDA runtime version: %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
  std::printf("Device count: %d\n", device_count);

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, i);
    if (err != cudaSuccess) {
      die("cudaGetDeviceProperties", err);
    }

    const double total_mem_gib = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

    std::printf("\n--- Device %d ---\n", i);
    std::printf("name: %s\n", prop.name);
    std::printf("compute_capability: %d.%d\n", prop.major, prop.minor);
    std::printf("total_global_mem_GiB: %.3f\n", total_mem_gib);
    std::printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
    std::printf("warpSize: %d\n", prop.warpSize);
    std::printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    std::printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("regsPerBlock: %d\n", prop.regsPerBlock);
    std::printf("regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
    std::printf("sharedMemPerBlock: %zu\n", prop.sharedMemPerBlock);
    std::printf("sharedMemPerMultiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
    // CUDA 13+ removed clockRate / memoryClockRate from cudaDeviceProp.
    std::printf("memoryBusWidth_bits: %d\n", prop.memoryBusWidth);
    std::printf("l2CacheSize: %d\n", prop.l2CacheSize);
    std::printf("concurrentKernels: %d\n", prop.concurrentKernels);
  }

  return 0;
}
