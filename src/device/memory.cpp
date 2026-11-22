#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>

#include "device/memory.h"
#include "utils/logger.h"

namespace mtensor {

std::shared_ptr<void> AllocateStorage(std::size_t bytes, Device device) {
    if (device == Device::kCPU) {
        return std::shared_ptr<void>(AllocateCPU(bytes), std::free);
    }
    if (device == Device::kCUDA) {
#ifdef ENABLE_CUDA
        return std::shared_ptr<void>(AllocateCUDA(bytes), FreeCUDA);
#else
        throw std::logic_error("AllocateStorage: CUDA support is disabled");
#endif
    }
    throw std::invalid_argument("AllocateStorage: unsupported device");
}

void* AllocateCPU(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }

    void* host_ptr_ = std::malloc(bytes);
    if (host_ptr_ == nullptr) {
        throw std::bad_alloc{};
    }
    return host_ptr_;
}

#ifdef ENABLE_CUDA
void* AllocateCUDA(std::size_t bytes){
    if(bytes == 0)
        return nullptr;

    void* device_ptr_ = nullptr;
    CUDA_CHECK(cudaMalloc(&device_ptr_, bytes));
    return device_ptr_;
}

void FreeCUDA(void* device_ptr_) noexcept {
    if (device_ptr_ == nullptr) {
        return;
    }

    const cudaError_t status = cudaFree(device_ptr_);
    if (status == cudaSuccess) {
        return;
    }

    try {
        logger.error(std::string("cudaFree failed: ") + cudaGetErrorString(status));
    } catch (...) {
        std::fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(status));
    }
}
#endif

}  // namespace mtensor
