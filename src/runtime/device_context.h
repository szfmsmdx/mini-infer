#pragma once

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/tensor/device.h"

namespace mtensor {

// 持有设备执行期的流与同步状态；算子后端通过它访问流和后续临时空间。
class DeviceContext {
public:
    explicit DeviceContext(Device device) noexcept : device_(device) {}

    Device device() const noexcept { return device_; }
    void synchronize() const;

#ifdef ENABLE_CUDA
    cudaStream_t cuda_stream() const noexcept { return stream_; }
#endif

private:
    Device device_;
#ifdef ENABLE_CUDA
    cudaStream_t stream_ = nullptr;
#endif
};

DeviceContext& GetDeviceContext(Device device);

}  // namespace mtensor
