#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "ops/backend.h"

namespace mtensor {
namespace {

template <typename Input, typename Output>
void SqrtTypedCPU(const void* input, void* output, const std::vector<int>& shape,
                  const std::vector<std::size_t>& stride) {
    const auto* src = static_cast<const std::byte*>(input);
    auto* dst = static_cast<std::byte*>(output);
    auto visit = [&](auto&& self, std::size_t dim, std::size_t offset) -> void {
        if (dim == shape.size()) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const Output root(std::sqrt(static_cast<float>(value)));
            std::memcpy(dst, &root, sizeof(Output));
            dst += sizeof(Output);
            return;
        }
        for (int i = 0; i < shape[dim]; ++i) {
            self(self, dim + 1, offset + i * stride[dim]);
        }
    };
    visit(visit, 0, 0);
}

}  // namespace

void SqrtCPU(const void* input, void* output, Dtype dtype, const std::vector<int>& shape,
             const std::vector<std::size_t>& stride) {
    switch (dtype) {
        case Dtype::kFloat32: SqrtTypedCPU<float, float>(input, output, shape, stride); break;
        case Dtype::kFloat16: SqrtTypedCPU<Float16, Float16>(input, output, shape, stride); break;
        case Dtype::kBFloat16: SqrtTypedCPU<BFloat16, BFloat16>(input, output, shape, stride); break;
        case Dtype::kInt8: SqrtTypedCPU<std::int8_t, float>(input, output, shape, stride); break;
        case Dtype::kUInt8: SqrtTypedCPU<std::uint8_t, float>(input, output, shape, stride); break;
        case Dtype::kInt16: SqrtTypedCPU<std::int16_t, float>(input, output, shape, stride); break;
        case Dtype::kInt32: SqrtTypedCPU<std::int32_t, float>(input, output, shape, stride); break;
        case Dtype::kInt64: SqrtTypedCPU<std::int64_t, float>(input, output, shape, stride); break;
        default: throw std::invalid_argument("sqrt: unsupported dtype");
    }
}

}  // namespace mtensor
