#pragma once

#include <cstdint>
#include <stdexcept>

#include "mini_infer/tensor/tensor.h"

namespace mtensor_test {

inline void TestScalarArithmetic(mtensor::Device device) {
    using namespace mtensor;
    const float data[] = {2, 4};
    Tensor x(data, {2}, Dtype::kFloat32, device);
    x.requires_grad_(true);
    auto check = [&](const Tensor& result, float a, float b, const char* name) {
        if (result.shape() != x.shape() || result.dtype() != x.dtype() || result.device() != device)
            throw std::runtime_error(name);
        const Tensor host = result.to(Device::kCPU);
        const auto* p = static_cast<const float*>(host.data_ptr());
        if (p[0] != a || p[1] != b) throw std::runtime_error(name);
    };
    const Scalar half(0.5);
    check(x + half, 2.5f, 4.5f, "tensor + scalar");
    check(half + x, 2.5f, 4.5f, "scalar + tensor");
    check(x - 8, -6, -4, "tensor - scalar");
    check(8 - x, 6, 4, "scalar - tensor");
    check(x * half, 1, 2, "tensor * scalar");
    check(half * x, 1, 2, "scalar * tensor");
    check(x / 8, 0.25f, 0.5f, "tensor / scalar");
    check(8 / x, 4, 2, "scalar / tensor");
    (8 / x - x * half).sum().backward();
    check(x.grad(), -2.5f, -1, "scalar arithmetic backward");

    const auto large = (std::int64_t{1} << 60) + 3;
    const Tensor one = full({}, 1, Dtype::kInt64, device);
    const Tensor integer = (one + large).to(Device::kCPU);
    if (*static_cast<const std::int64_t*>(integer.data_ptr()) != large + 1)
        throw std::runtime_error("scalar arithmetic lost int64 precision");
    const Tensor truncated = (one + 1.75).to(Device::kCPU);
    if (*static_cast<const std::int64_t*>(truncated.data_ptr()) != 2)
        throw std::runtime_error("scalar arithmetic must use fill_ conversion");
}

}  // namespace mtensor_test
