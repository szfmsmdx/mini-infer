#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>

#include "utils/memory.h"
#include "utils/logger.h"

namespace mtensor {

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

#ifdef MINI_INFER_ENABLE_CUDA
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
