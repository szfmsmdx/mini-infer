#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "mini_infer/tensor/tensor.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "tensor/check.h"
#include "tensor/tensor_impl.h"

namespace mtensor {

Tensor& Tensor::fill_(const Scalar& value) {
    if (numel() == 0) {
        return *this;
    }
    CheckBroadcastWritable(*this);
    CheckDataPointer(*this, "Tensor::fill_: null data pointer");

    switch (dtype()) {
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

    Tensor saved_value = value;
    if (impl_->storage == value.impl_->storage) {
        NogradNode guard;
        saved_value = value.clone();
    }
    update_version();
    GetKernelTable().fill_tensor.call(device(), "Tensor::fill_", *this, saved_value);
    return *this;
}

Tensor& Tensor::FillValue(const void* host_value) {
    update_version();
    GetKernelTable().fill_value.call(device(), "Tensor::fill_", *this, host_value);
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
        update_version();
        GetKernelTable().zero.call(device(), "Tensor::zero_", *this);
    }
    return *this;
}

Tensor full(ShapeView shape, const Scalar& value, Dtype dtype, Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(value);
    return result;
}

Tensor full(ShapeView shape, const Tensor& value, Dtype dtype, Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(value);
    return result;
}

}  // namespace mtensor
