#pragma once

#include <cstdint>
#include <stdexcept>

#include "mini_infer/autograd.h"
#include "mini_infer/core/tensor.h"
#include "tensor/check.h"

namespace mtensor {

inline void CheckFloatMatrix(const Tensor& x, const char* message) {
    if (x.empty() || x.dim() != 2 || x.dtype() != Dtype::kFloat32) {
        throw std::invalid_argument(message);
    }
}

inline Tensor Packed(const Tensor& x) {
    NogradNode guard;
    return x.contiguous();
}

inline Tensor CheckAndPackIndices(const Tensor& indices, const Tensor& data, int upper_bound) {
    if (indices.empty() || indices.dim() != 1 || indices.dtype() != Dtype::kInt32) {
        throw std::invalid_argument("token ids must be a nonempty Int32 vector");
    }
    CheckSameDevice(indices, data, "token ids must be on the same device as the data");
    const Tensor packed = Packed(indices);
    const Tensor minimum = packed.min();
    const Tensor maximum = packed.max();
    const Tensor host_min = minimum.to(Device::kCPU);
    const Tensor host_max = maximum.to(Device::kCPU);
    if (*static_cast<const std::int32_t*>(host_min.data_ptr()) < 0 ||
        *static_cast<const std::int32_t*>(host_max.data_ptr()) >= upper_bound) {
        throw std::out_of_range("token id is outside the vocabulary");
    }
    return packed;
}

}  // namespace mtensor
