#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "autograd/backend.h"
#include "mini_infer/tensor.h"

namespace mtensor {
namespace {

template <typename T>
void RsqrtBackwardTyped(const Tensor& x, const Tensor& grad, Tensor& dx) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(dx.data_ptr());
    const auto* g = static_cast<const std::byte*>(grad.data_ptr());
    auto write = [&](std::size_t dst_index, std::size_t x_index, std::size_t g_index) {
        T value(0);
        std::memcpy(&value, src + x_index * sizeof(T), sizeof(T));
        const float y = 1.0f / std::sqrt(static_cast<float>(value));
        T upstream(0);
        std::memcpy(&upstream, g + g_index * sizeof(T), sizeof(T));
        const T out((-0.5f * y * static_cast<float>(upstream)) * y * y);
        std::memcpy(dst + dst_index * sizeof(T), &out, sizeof(T));
    };
    if (x.is_contiguous() && grad.is_contiguous()) {
        const auto count = x.numel();
        for (std::size_t i = 0; i < count; ++i) write(i, i, i);
        return;
    }
    std::size_t index = 0;
    auto visit = [&](auto&& self, int dim, std::size_t xi, std::size_t gi) -> void {
        if (dim == x.dim()) {
            write(index++, xi, gi);
            return;
        }
        const auto xs = x.stride()[dim];
        const auto gs = grad.stride()[dim];
        for (int i = 0; i < x.size(dim); ++i) self(self, dim + 1, xi + i * xs, gi + i * gs);
    };
    visit(visit, 0, 0, 0);
}

}  // namespace

void RsqrtBackwardCPU(const Tensor& x, const Tensor& grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: RsqrtBackwardTyped<float>(x, grad, dx); break;
        case Dtype::kFloat16: RsqrtBackwardTyped<Float16>(x, grad, dx); break;
        case Dtype::kBFloat16: RsqrtBackwardTyped<BFloat16>(x, grad, dx); break;
        default: throw std::invalid_argument("rsqrt backward: unsupported dtype");
    }
}

}  // namespace mtensor
