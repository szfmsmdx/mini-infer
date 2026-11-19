#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "ops/backend.h"
#include "mini_infer/tensor.h"

namespace mtensor {
namespace {

template <typename In, typename Out>
void RsqrtTypedCPU(const Tensor& x, Tensor& result) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(result.data_ptr());
    auto write = [&](std::size_t dst_index, std::size_t x_index) {
        In value(0);
        std::memcpy(&value, src + x_index * sizeof(In), sizeof(In));
        const float y = 1.0f / std::sqrt(static_cast<float>(value));
        const Out out(y);
        std::memcpy(dst + dst_index * sizeof(Out), &out, sizeof(Out));
    };
    if (x.is_contiguous()) {
        const auto count = x.numel();
        for (std::size_t i = 0; i < count; ++i) write(i, i);
        return;
    }
    std::size_t index = 0;
    auto visit = [&](auto&& self, int dim, std::size_t xi) -> void {
        if (dim == x.dim()) {
            write(index++, xi);
            return;
        }
        const auto xs = x.stride()[dim];
        for (int i = 0; i < x.size(dim); ++i) self(self, dim + 1, xi + i * xs);
    };
    visit(visit, 0, 0);
}

}  // namespace

void RsqrtCPU(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: RsqrtTypedCPU<float, float>(x, out); break;
        case Dtype::kFloat16: RsqrtTypedCPU<Float16, Float16>(x, out); break;
        case Dtype::kBFloat16: RsqrtTypedCPU<BFloat16, BFloat16>(x, out); break;
        case Dtype::kInt8: RsqrtTypedCPU<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: RsqrtTypedCPU<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: RsqrtTypedCPU<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: RsqrtTypedCPU<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: RsqrtTypedCPU<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("rsqrt: unsupported dtype");
    }
}

}  // namespace mtensor
