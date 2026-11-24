#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include "device/memory.h"
#include "device/runtime.h"
#include "utils/logger.h"

namespace mtensor {
namespace {

void FreeCPU(void* data) noexcept { std::free(data); }

class CPUAllocator : public Allocator {
public:
    DataPtr allocate(std::size_t bytes) const override {
        if (!bytes) return {nullptr, Device::kCPU, FreeCPU};
        auto* data = std::malloc(bytes);
        if (!data) throw std::bad_alloc{};
        return {data, Device::kCPU, FreeCPU};
    }
};

#ifdef ENABLE_CUDA
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

class CUDAAllocator : public Allocator {
public:
    DataPtr allocate(std::size_t bytes) const override {
        if (!bytes) return {nullptr, Device::kCUDA, FreeCUDA};
        void* data = nullptr;
        CUDA_CHECK(cudaMalloc(&data, bytes));
        return {data, Device::kCUDA, FreeCUDA};
    }
};
#endif

}  // namespace

const Allocator& GetAllocator(Device device) {
    if (device == Device::kCPU) {
        static const CPUAllocator allocator;
        return allocator;
    }
    if (device == Device::kCUDA) {
#ifdef ENABLE_CUDA
        static const CUDAAllocator allocator;
        return allocator;
#else
        throw std::logic_error("GetAllocator: CUDA support is disabled");
#endif
    }
    throw std::invalid_argument("GetAllocator: unsupported device");
}

void CopyBytes(void* dst, Device dst_device, const void* src, Device src_device,
               std::size_t bytes) {
    if (!bytes) return;
    if (dst_device == Device::kCPU && src_device == Device::kCPU) {
        std::memcpy(dst, src, bytes);
        return;
    }
#ifdef ENABLE_CUDA
    const auto kind = dst_device == Device::kCPU ? cudaMemcpyDeviceToHost :
                      src_device == Device::kCPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
#else
    throw std::logic_error("CopyBytes: CUDA support is disabled");
#endif
}

}  // namespace mtensor
