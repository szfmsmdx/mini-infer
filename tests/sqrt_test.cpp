#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef SQRT_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/autograd.h"
#include "mini_infer/ops.h"

namespace {
using namespace mtensor;

#ifdef SQRT_TEST_CUDA
constexpr Device kDevice = Device::kCUDA;
#else
constexpr Device kDevice = Device::kCPU;
#endif

void Require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template <typename T>
void CheckValues(const Tensor& result, const std::vector<int>& shape, Dtype dtype,
                 const std::vector<float>& expected, const char* message, float tolerance = 0.0f) {
    Require(result.shape() == shape && result.dtype() == dtype && result.device() == kDevice &&
            result.numel() == expected.size(), message);
    const Tensor host = result.to(Device::kCPU).contiguous();
    const auto* data = static_cast<const T*>(host.data_ptr());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        Require(std::abs(static_cast<float>(data[i]) - expected[i]) <= tolerance, message);
    }
}

template <typename T>
void CheckInteger(Dtype dtype) {
    const T data[] = {2, 9};
    Tensor x(data, {2}, dtype, kDevice);
    x.requires_grad_(true);
    const Tensor root = sqrt(x);
    const Tensor inverse = rsqrt(x);
    Require(!root.grad_node() && !inverse.grad_node(), "integer input recorded a grad node");
    CheckValues<float>(root, {2}, Dtype::kFloat32, {std::sqrt(2.0f), 3}, "integer sqrt");
    CheckValues<float>(inverse, {2}, Dtype::kFloat32, {1 / std::sqrt(2.0f), 1 / 3.0f},
                       "integer rsqrt", std::numeric_limits<float>::epsilon());
}

template <typename T>
void CheckFloating(Dtype dtype) {
    const T data[] = {T(1), T(4), T(16), T(64)};
    const T upstream[] = {T(2), T(4), T(8), T(16)};
    Tensor x(data, {2, 2}, dtype, kDevice);
    x.requires_grad_(true);
    const Tensor view = x.transpose(0, 1);
    const Tensor g(upstream, {2, 2}, dtype, kDevice);
    Tensor y = view.sqrt();
    CheckValues<T>(y, {2, 2}, dtype, {1, 4, 2, 8}, "sqrt strided view");
    y.backward(g);
    CheckValues<T>(x.grad(), {2, 2}, dtype, {1, 2, 0.5f, 1}, "sqrt backward");
    x.zero_grad();
    Tensor z = view.rsqrt();
    CheckValues<T>(z, {2, 2}, dtype, {1, 0.25f, 0.5f, 0.125f}, "rsqrt strided view");
    z.backward(g);
    CheckValues<T>(x.grad(), {2, 2}, dtype, {-1, -0.5f, -0.03125f, -0.015625f}, "rsqrt backward");

    x.zero_();
    for (Tensor* output : {&y, &z}) {
        try {
            output->backward(g);
        } catch (const std::logic_error&) {
            continue;
        }
        throw std::runtime_error("sqrt/rsqrt accepted a modified input");
    }
}

void CheckRsqrtSingleRounding() {
    // 如果先将 sqrt 写回低精度，再取倒数，这两个输入会得到不同的舍入结果。
    const Float16 a(17);
    const BFloat16 b(11);
    CheckValues<Float16>(rsqrt(Tensor(&a, {}, Dtype::kFloat16, kDevice)), {}, Dtype::kFloat16,
                         {0.2425537109375f}, "FP16 rsqrt rounded twice");
    CheckValues<BFloat16>(rsqrt(Tensor(&b, {}, Dtype::kBFloat16, kDevice)), {}, Dtype::kBFloat16,
                          {0.30078125f}, "BF16 rsqrt rounded twice");
}

void CheckSpecialValues() {
    const float data[] = {0.0f, -0.0f, -1, std::numeric_limits<float>::infinity()};
    const Tensor x(data, {4}, Dtype::kFloat32, kDevice);
    NogradNode guard;
    const Tensor root = sqrt(x).to(Device::kCPU);
    const Tensor inverse = rsqrt(x).to(Device::kCPU);
    Require(!root.grad_node() && !inverse.grad_node(), "no-grad recorded a node");
    const auto* y = static_cast<const float*>(root.data_ptr());
    const auto* z = static_cast<const float*>(inverse.data_ptr());
    Require(y[0] == 0 && !std::signbit(y[0]) && y[1] == 0 && std::signbit(y[1]) &&
            std::isnan(y[2]) && y[3] == data[3], "sqrt special values");
    Require(z[0] == data[3] && z[1] == -data[3] && std::isnan(z[2]) && z[3] == 0, "rsqrt special values");
}
}  // namespace

int main() {
    try {
#ifdef SQRT_TEST_CUDA
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
            (status == cudaSuccess && count == 0)) return 77;
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
        CheckInteger<std::int8_t>(Dtype::kInt8);
        CheckInteger<std::uint8_t>(Dtype::kUInt8);
        CheckInteger<std::int16_t>(Dtype::kInt16);
        CheckInteger<std::int32_t>(Dtype::kInt32);
        CheckInteger<std::int64_t>(Dtype::kInt64);
        CheckFloating<float>(Dtype::kFloat32);
        CheckFloating<Float16>(Dtype::kFloat16);
        CheckFloating<BFloat16>(Dtype::kBFloat16);
        CheckSpecialValues();
        CheckRsqrtSingleRounding();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
