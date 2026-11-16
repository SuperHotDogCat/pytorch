#pragma once

#include <ATen/cuda/Exceptions.h>

#include <cuda.h>
#include <cuda_runtime.h>
namespace extension {

  TORCH_API void registerNUMAPointer(void* ptr, size_t bytes);
  TORCH_API void unregisterNUMAPointer(void* ptr);
  TORCH_API bool isNUMAPointer(void* ptr);

}

namespace at::cuda {

inline Device getDeviceFromPtr(void* ptr) {
  cudaPointerAttributes attr{};

  AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  if (extension::isNUMAPointer(ptr)){
    return {c10::DeviceType::CUDA, 0}; // device_index = 0としておく
  }

#if !defined(USE_ROCM)
  TORCH_CHECK(attr.type != cudaMemoryTypeUnregistered,
    "The specified pointer resides on host memory and is not registered with any CUDA device.");
#endif

  return {c10::DeviceType::CUDA, static_cast<DeviceIndex>(attr.device)};
}

} // namespace at::cuda
