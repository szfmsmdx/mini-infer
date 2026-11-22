#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#ifdef LOG_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/ops/elementwise.h"

namespace {
using namespace mtensor;

template <typename In, typename Out>
void CheckForward(Dtype input_dtype, Dtype output_dtype, Device device) {
    const In values[] = {In(1), In(4), In(2), In(8)};
    const Tensor x(values, {2, 2}, input_dtype, device);
    const Tensor y = x.transpose(0, 1).log();
    if (y.shape() != x.shape() || y.dtype() != output_dtype || y.device() != device)
        throw std::runtime_error("log: incorrect output metadata");

    const Tensor host = y.to(Device::kCPU);
    const auto* actual = static_cast<const Out*>(host.data_ptr());
    const float inputs[] = {1, 2, 4, 8};
    for (int i = 0; i < 4; ++i) {
        const Out rounded(std::log(inputs[i]));
        if (!(std::abs(static_cast<float>(actual[i]) - static_cast<float>(rounded)) <= 1e-6f))
            throw std::runtime_error("log: incorrect value for transposed input");
    }
}

void CheckSpecialValues(Device device) {
    const float inf = std::numeric_limits<float>::infinity();
    const float values[] = {-1.0f, -0.0f, 0.0f, inf};
    const Tensor x(values, {4}, Dtype::kFloat32, device);
    const Tensor host = log(x).to(Device::kCPU);
    const auto* p = static_cast<const float*>(host.data_ptr());
    if (!std::isnan(p[0]) || p[1] != -inf || p[2] != -inf || p[3] != inf)
        throw std::runtime_error("log: wrong result for negative, zero or infinite input");
}

template <typename T>
void CheckBackward(Device device, Dtype dtype) {
    const T values[] = {T(3), T(1), T(7), T(1)};
    const T upstream[] = {T(2), T(-5)};
    Tensor x(values, {2, 2}, dtype, device);
    x.requires_grad_(true);
    const Tensor view = x.narrow(1, 0, 1).squeeze(1);
    Tensor y = view.log();
    const Tensor grad(upstream, {2}, dtype, device);
    y.backward(grad);
    const Tensor dx = x.grad();
    if (dx.dtype() != dtype || dx.device() != device || dx.shape() != x.shape())
        throw std::runtime_error("log backward: incorrect gradient metadata");
    const Tensor host = dx.to(Device::kCPU);
    const auto* p = static_cast<const T*>(host.data_ptr());
    const T expected[] = {T(2.0f / 3.0f), T(0), T(-5.0f / 7.0f), T(0)};
    for (int i = 0; i < 4; ++i) {
        if (static_cast<float>(p[i]) != static_cast<float>(expected[i]))
            throw std::runtime_error("log backward: incorrect gradient");
    }

    x.zero_();
    try {
        y.backward(grad);
    } catch (const std::logic_error&) {
        return;
    }
    throw std::runtime_error("log backward accepted a modified input");
}
}  // namespace

int main() {
    using namespace mtensor;
#ifdef LOG_TEST_CUDA
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
        (status == cudaSuccess && count == 0)) return 77;
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    constexpr auto device = Device::kCUDA;
#else
    constexpr auto device = Device::kCPU;
#endif
    CheckSpecialValues(device);
    CheckBackward<float>(device, Dtype::kFloat32);
    CheckBackward<Float16>(device, Dtype::kFloat16);
    CheckBackward<BFloat16>(device, Dtype::kBFloat16);
    CheckForward<float, float>(Dtype::kFloat32, Dtype::kFloat32, device);
    CheckForward<Float16, Float16>(Dtype::kFloat16, Dtype::kFloat16, device);
    CheckForward<BFloat16, BFloat16>(Dtype::kBFloat16, Dtype::kBFloat16, device);
    CheckForward<std::int8_t, float>(Dtype::kInt8, Dtype::kFloat32, device);
    CheckForward<std::uint8_t, float>(Dtype::kUInt8, Dtype::kFloat32, device);
    CheckForward<std::int16_t, float>(Dtype::kInt16, Dtype::kFloat32, device);
    CheckForward<std::int32_t, float>(Dtype::kInt32, Dtype::kFloat32, device);
    CheckForward<std::int64_t, float>(Dtype::kInt64, Dtype::kFloat32, device);
}
