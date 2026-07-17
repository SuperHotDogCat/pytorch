#pragma once

#include <ATen/cuda/Exceptions.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace at::cuda {

inline Device getDeviceFromPtr(void* ptr) {
  cudaPointerAttributes attr{};

  AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));

  if (attr.type == cudaMemoryTypeUnregistered){
    void *p; // For GH200 Unified Virtual Address build
    AT_CUDA_CHECK(cudaMalloc(&p, 1));
    AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, p));
    AT_CUDA_CHECK(cudaFree(p));
  }
#if !defined(USE_ROCM)
  TORCH_CHECK(attr.type != cudaMemoryTypeUnregistered,
    "The specified pointer resides on host memory and is not registered with any CUDA device.");
#endif

  return {c10::DeviceType::CUDA, static_cast<DeviceIndex>(attr.device)};
}

} // namespace at::cuda
