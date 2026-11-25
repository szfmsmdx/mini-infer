#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

void CrossEntropyRowsCPU(const Tensor& logits, const Tensor& targets, Tensor& row_losses) {
    const auto* values = static_cast<const float*>(logits.data_ptr());
    const auto* ids = static_cast<const std::int32_t*>(targets.data_ptr());
    auto* result = static_cast<float*>(row_losses.data_ptr());
    const int rows = logits.size(0);
    const int cols = logits.size(1);
    for (int row = 0; row < rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < cols; ++col) {
            maximum = std::max(maximum, values[
                static_cast<std::size_t>(row) * logits.stride()[0] +
                static_cast<std::size_t>(col) * logits.stride()[1]]);
        }
        float partition = 0.0f;
        for (int col = 0; col < cols; ++col) {
            partition += std::exp(values[
                static_cast<std::size_t>(row) * logits.stride()[0] +
                static_cast<std::size_t>(col) * logits.stride()[1]] - maximum);
        }
        const float target = values[
            static_cast<std::size_t>(row) * logits.stride()[0] +
            static_cast<std::size_t>(ids[row]) * logits.stride()[1]];
        result[row] = (maximum - target) + std::log(partition);
    }
}

void CrossEntropyGradCPU(const Tensor& logits, const Tensor& targets,
                         const Tensor& grad, Tensor& logits_grad) {
    const auto* values = static_cast<const float*>(logits.data_ptr());
    const auto* ids = static_cast<const std::int32_t*>(targets.data_ptr());
    const float upstream = *static_cast<const float*>(grad.data_ptr()) / logits.size(0);
    auto* result = static_cast<float*>(logits_grad.data_ptr());
    const int rows = logits.size(0);
    const int cols = logits.size(1);
    for (int row = 0; row < rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < cols; ++col) {
            maximum = std::max(maximum, values[
                static_cast<std::size_t>(row) * logits.stride()[0] +
                static_cast<std::size_t>(col) * logits.stride()[1]]);
        }
        float partition = 0.0f;
        for (int col = 0; col < cols; ++col) {
            partition += std::exp(values[
                static_cast<std::size_t>(row) * logits.stride()[0] +
                static_cast<std::size_t>(col) * logits.stride()[1]] - maximum);
        }
        for (int col = 0; col < cols; ++col) {
            const float value = values[
                static_cast<std::size_t>(row) * logits.stride()[0] +
                static_cast<std::size_t>(col) * logits.stride()[1]];
            const float probability = std::exp(value - maximum) / partition;
            result[static_cast<std::size_t>(row) * cols + col] =
                (probability - (col == ids[row] ? 1.0f : 0.0f)) * upstream;
        }
    }
}

}  // namespace mtensor
