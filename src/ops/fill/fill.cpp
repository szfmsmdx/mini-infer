#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "mini_infer/tensor.h"
#include "mini_infer/autograd.h"
#include "tensor/check.h"
#include "ops/fill/kernels.h"

namespace mtensor {

Tensor& Tensor::fill_(const Scalar& value) {
    if (numel() == 0) {
        return *this;
    }
    CheckBroadcastWritable(*this);
    CheckDataPointer(*this, "Tensor::fill_: null data pointer");

    switch (dtype_) {
        case Dtype::kFloat32: {
            const float converted = value.to<float>();
            return FillValue(&converted);
        }
        case Dtype::kFloat16: {
            const Float16 converted(value.to<float>());
            return FillValue(&converted.bits);
        }
        case Dtype::kBFloat16: {
            const BFloat16 converted(value.to<float>());
            return FillValue(&converted.bits);
        }
        case Dtype::kInt8: {
            const std::int8_t converted = value.to<std::int8_t>();
            return FillValue(&converted);
        }
        case Dtype::kUInt8: {
            const std::uint8_t converted = value.to<std::uint8_t>();
            return FillValue(&converted);
        }
        case Dtype::kInt16: {
            const std::int16_t converted = value.to<std::int16_t>();
            return FillValue(&converted);
        }
        case Dtype::kInt32: {
            const std::int32_t converted = value.to<std::int32_t>();
            return FillValue(&converted);
        }
        case Dtype::kInt64: {
            const std::int64_t converted = value.to<std::int64_t>();
            return FillValue(&converted);
        }
        default: throw std::invalid_argument("unsupported dtype");
    }
}

Tensor& Tensor::fill_(const Tensor& value) {
    if (!value.is_scalar()) {
        throw std::invalid_argument("Tensor::fill_: value must be a non-empty scalar Tensor");
    }
    CheckSameDtype(*this, value, "Tensor::fill_: dtype and device must match");
    CheckSameDevice(*this, value, "Tensor::fill_: dtype and device must match");
    if (empty()) {
        return *this;
    }
    CheckBroadcastWritable(*this);

    if (device_ == Device::kCPU) {
        // 先保存填充值，避免 value 与目标共享存储时被覆盖。
        std::vector<std::byte> saved_value(element_size());
        std::memcpy(saved_value.data(), value.data_ptr_, element_size());
        update_version();
        FillCPU(data_ptr_, shape_, stride_, element_size(), saved_value.data());
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        if (storage_ == value.storage_) {
            NogradNode guard;
            Tensor saved_value = value.clone();
            update_version();
            FillCUDAFromDevice(data_ptr_, numel(), element_size(), saved_value.data_ptr_,
                               shape_cuda_.get(), stride_cuda_.get(), dim());
        } else {
            update_version();
            FillCUDAFromDevice(data_ptr_, numel(), element_size(), value.data_ptr_,
                               shape_cuda_.get(), stride_cuda_.get(), dim());
        }
#else
        throw std::logic_error("Tensor::fill_: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::fill_: unsupported device");
    }
    return *this;
}

Tensor& Tensor::FillValue(const void* host_value) {
    if (device_ == Device::kCPU) {
        update_version();
        FillCPU(data_ptr_, shape_, stride_, element_size(), host_value);
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        update_version();
        FillCUDA(data_ptr_, numel(), dtype_, host_value, shape_cuda_.get(), stride_cuda_.get(), dim());
#else
        throw std::logic_error("Tensor::fill_: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::fill_: unsupported device");
    }
    return *this;
}

Tensor& Tensor::zero_() {
    /*
    使用示例（a 为已构造的 CPU Tensor）：
        a.zero_();
    */
    const std::size_t bytes = nbytes();
    if(bytes > 0){
        CheckBroadcastWritable(*this);
        if (device_ == Device::kCPU) {
            update_version();
            ZeroCPU(data_ptr_, shape_, stride_, element_size());
        } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
            update_version();
            ZeroCUDA(data_ptr_, shape_cuda_.get(), stride_cuda_.get(),
                     dim(), numel(), element_size());
#else
            throw std::logic_error("Tensor::zero_: CUDA support is disabled");
#endif
        } else {
            throw std::invalid_argument("Tensor::zero_: unsupported device");
        }
    }
    return *this;
}

Tensor full(const std::vector<int>& shape, const Scalar& value, Dtype dtype, Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(value);
    return result;
}

Tensor full(const std::vector<int>& shape, const Tensor& value, Dtype dtype, Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(value);
    return result;
}

}  // namespace mtensor
