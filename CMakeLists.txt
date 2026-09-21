cmake_minimum_required(VERSION 3.24)

project(qkern LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Prefer an explicit toolkit root when CUDA_PATH is mis-set on the machine.
if(DEFINED ENV{CUDAToolkit_ROOT})
  set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}" CACHE PATH "" FORCE)
elseif(EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
  set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" CACHE PATH "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# GTX 1650 is Turing (sm_75). CMake will embed this arch for Phase 0 smoke builds.
# Later phases should prefer runtime arch detection / fat binaries where practical.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(qkern_device_query
  src/tools/device_query.cu
)
target_link_libraries(qkern_device_query PRIVATE CUDA::cudart)
set_target_properties(qkern_device_query PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

add_executable(qkern_nvrtc_smoke
  src/tools/nvrtc_smoke.cpp
)
target_include_directories(qkern_nvrtc_smoke PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(qkern_nvrtc_smoke PRIVATE CUDA::nvrtc CUDA::cudart)
set_target_properties(qkern_nvrtc_smoke PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

message(STATUS "QKern: CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
message(STATUS "QKern: CUDA architectures=${CMAKE_CUDA_ARCHITECTURES}")

# FP16 GEMV variants + INT8/INT4 fused GEMV.
add_library(qkern_fp16_gemv STATIC
  src/kernels/fp16_gemv_naive.cu
  src/kernels/fp16_gemv_x_smem.cu
  src/kernels/fp16_gemv_vec2.cu
  src/kernels/fp16_gemv_dispatch.cpp
  src/kernels/int8_gemv.cu
  src/kernels/int4_gemv.cu
)
target_include_directories(qkern_fp16_gemv PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qkern_fp16_gemv PUBLIC CUDA::cudart)
set_target_properties(qkern_fp16_gemv PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)
