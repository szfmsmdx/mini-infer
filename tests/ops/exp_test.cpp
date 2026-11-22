#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#ifdef EXP_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/ops/elementwise.h"

namespace {
using namespace mtensor;

template <typename In, typename Out>
void Check(Dtype input_dtype, Dtype output_dtype, Device device, int first) {
    const In values[] = {In(first), In(0), In(2), In(1)};
    Tensor x(values, {2, 2}, input_dtype, device);
    Tensor y = x.transpose(0, 1).exp();
    if (y.shape() != x.shape() || y.dtype() != output_dtype || y.device() != device)
        throw std::runtime_error("exp: incorrect output metadata");

    Tensor host = y.to(Device::kCPU);
    const auto* actual = static_cast<const Out*>(host.data_ptr());
    const int exponents[] = {first, 2, 0, 1};
    for (int i = 0; i < 4; ++i) {
        float expected = static_cast<float>(Out(std::exp(static_cast<float>(exponents[i]))));
        if (!(std::abs(static_cast<float>(actual[i]) - expected) <= 1e-6f * expected))
            throw std::runtime_error("exp: incorrect value for transposed input");
    }
}

void CheckBlockBoundary(Device device) {
    constexpr int n = 513;
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i) values[i] = static_cast<float>(i % 7 - 3);
    const Tensor host = Tensor(values.data(), {n}, Dtype::kFloat32, device).exp().to(Device::kCPU);
    const auto* p = static_cast<const float*>(host.data_ptr());
    for (int i = 0; i < n; ++i) {
        const float expected = std::exp(values[i]);
        if (!(std::abs(p[i] - expected) <= 1e-6f * expected))
            throw std::runtime_error("exp: incorrect value across block boundary");
    }
}

void CheckBackward(Device device) {
    const float values[] = {0, 1}, upstream[] = {2, 3};
    Tensor x(values, {2}, Dtype::kFloat32, device);
    x.requires_grad_(true);
    Tensor y = x.exp();
    const Tensor g(upstream, {2}, Dtype::kFloat32, device);
    x.zero_();  // 反向使用前向的 y，不再读取 x 的值。
    y.backward(g);
    const Tensor dx = x.grad().to(Device::kCPU);
    const auto* p = static_cast<const float*>(dx.data_ptr());
    if (!(p[0] == 2 && std::abs(p[1] - 3 * std::exp(1.0f)) < 1e-5f))
        throw std::runtime_error("exp backward: incorrect gradient");

    y.zero_();
    try {
        y.backward(g);
    } catch (const std::logic_error&) {
        return;
    }
    throw std::runtime_error("exp backward accepted a modified output");
}
}  // namespace

int main() {
    using namespace mtensor;
#ifdef EXP_TEST_CUDA
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
        (status == cudaSuccess && count == 0)) return 77;
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    constexpr auto device = Device::kCUDA;
#else
    constexpr auto device = Device::kCPU;
#endif
    CheckBlockBoundary(device);
    CheckBackward(device);
    Check<float, float>(Dtype::kFloat32, Dtype::kFloat32, device, -1);
    Check<Float16, Float16>(Dtype::kFloat16, Dtype::kFloat16, device, -1);
    Check<BFloat16, BFloat16>(Dtype::kBFloat16, Dtype::kBFloat16, device, -1);
    Check<std::int8_t, float>(Dtype::kInt8, Dtype::kFloat32, device, -1);
    Check<std::uint8_t, float>(Dtype::kUInt8, Dtype::kFloat32, device, 3);
    Check<std::int16_t, float>(Dtype::kInt16, Dtype::kFloat32, device, -1);
    Check<std::int32_t, float>(Dtype::kInt32, Dtype::kFloat32, device, -1);
    Check<std::int64_t, float>(Dtype::kInt64, Dtype::kFloat32, device, -1);
}
