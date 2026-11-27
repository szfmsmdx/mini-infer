#include <cstdint>

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

void EmbeddingCPU(const Tensor& weight, const Tensor& indices, Tensor& output) {
    const auto* ids = static_cast<const std::int32_t*>(indices.data_ptr());
    const auto* values = static_cast<const float*>(weight.data_ptr());
    auto* result = static_cast<float*>(output.data_ptr());
    const int width = weight.size(1);
    for (int row = 0; row < indices.size(0); ++row) {
        const std::size_t source_row = static_cast<std::size_t>(ids[row]) * weight.stride()[0];
        for (int col = 0; col < width; ++col) {
            result[static_cast<std::size_t>(row) * width + col] =
                values[source_row + static_cast<std::size_t>(col) * weight.stride()[1]];
        }
    }
}

void EmbeddingGradCPU(const Tensor& grad, const Tensor& indices, Tensor& weight_grad) {
    const auto* ids = static_cast<const std::int32_t*>(indices.data_ptr());
    const auto* source = static_cast<const float*>(grad.data_ptr());
    auto* result = static_cast<float*>(weight_grad.data_ptr());
    const int width = weight_grad.size(1);
    for (int row = 0; row < indices.size(0); ++row) {
        const std::size_t target_row = static_cast<std::size_t>(ids[row]) * width;
        for (int col = 0; col < width; ++col) {
            result[target_row + col] += source[
                static_cast<std::size_t>(row) * grad.stride()[0] +
                static_cast<std::size_t>(col) * grad.stride()[1]];
        }
    }
}

}  // namespace mtensor
