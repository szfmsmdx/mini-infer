#include <stdexcept>

#include "runtime/device_context.h"
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "runtime/cuda_check.h"
#endif

namespace mtensor {

void DeviceContext::synchronize() const {
    if (device_ == Device::kCPU) {
        return;
    }
    if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return;
#else
        throw std::logic_error("DeviceContext::synchronize: CUDA support is disabled");
#endif
    }
    throw std::invalid_argument("DeviceContext::synchronize: unsupported device");
}

DeviceContext& GetDeviceContext(Device device) {
    if (device == Device::kCPU) {
        static DeviceContext cpu_ctx(Device::kCPU);
        return cpu_ctx;
    }
    if (device == Device::kCUDA) {
#ifdef ENABLE_CUDA
        static DeviceContext cuda_ctx(Device::kCUDA);
        return cuda_ctx;
#else
        throw std::logic_error("GetDeviceContext: CUDA support is disabled");
#endif
    }
    throw std::invalid_argument("GetDeviceContext: unsupported device");
}

}  // namespace mtensor
