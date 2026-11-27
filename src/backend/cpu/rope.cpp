#include <cmath>
#include <cstddef>

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

void RoPECPU(const Tensor& x, int head_width, float base, bool inverse,
             int pos_offset, Tensor& output) {
    const int rows = x.size(0);
    const int width = x.size(1);
    const int pairs_per_head = head_width / 2;
    const auto* input = static_cast<const float*>(x.data_ptr());
    auto* result = static_cast<float*>(output.data_ptr());
    for (int row = 0; row < rows; ++row) {
        const float position = static_cast<float>(
            static_cast<std::size_t>(pos_offset) + static_cast<std::size_t>(row));
        for (int head_col = 0; head_col < width; head_col += head_width) {
            for (int pair = 0; pair < pairs_per_head; ++pair) {
                const float exponent = -static_cast<float>(2 * pair) / head_width;
                const float angle = position * std::pow(base, exponent);
                const float cosine = std::cos(angle);
                const float sine = inverse ? -std::sin(angle) : std::sin(angle);
                const std::size_t index = static_cast<std::size_t>(row) * width +
                    head_col + 2 * pair;
                const float first = input[index];
                const float second = input[index + 1];
                result[index] = first * cosine - second * sine;
                result[index + 1] = first * sine + second * cosine;
            }
        }
    }
}

}  // namespace mtensor
