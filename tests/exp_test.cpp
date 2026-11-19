#include <cmath>
#include <cstdint>
#include <stdexcept>

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
}  // namespace

int main() {
    using namespace mtensor;
    constexpr auto device = Device::kCPU;
    Check<float, float>(Dtype::kFloat32, Dtype::kFloat32, device, -1);
    Check<Float16, Float16>(Dtype::kFloat16, Dtype::kFloat16, device, -1);
    Check<BFloat16, BFloat16>(Dtype::kBFloat16, Dtype::kBFloat16, device, -1);
    Check<std::int8_t, float>(Dtype::kInt8, Dtype::kFloat32, device, -1);
    Check<std::uint8_t, float>(Dtype::kUInt8, Dtype::kFloat32, device, 3);
    Check<std::int16_t, float>(Dtype::kInt16, Dtype::kFloat32, device, -1);
    Check<std::int32_t, float>(Dtype::kInt32, Dtype::kFloat32, device, -1);
    Check<std::int64_t, float>(Dtype::kInt64, Dtype::kFloat32, device, -1);
}
