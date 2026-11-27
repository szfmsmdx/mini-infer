#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "mini_infer/tensor/tensor.h"
#include "backend/common/dtype_dispatch.h"

namespace mtensor {
namespace {

std::size_t OffsetAt(const Tensor& x, std::size_t index) {
    std::size_t offset = 0;
    for (int dim = x.dim() - 1; dim >= 0; --dim) {
        offset += index % static_cast<std::size_t>(x.size(dim)) * x.stride()[dim];
        index /= static_cast<std::size_t>(x.size(dim));
    }
    return offset;
}

template <typename T>
void ForwardTyped(const Tensor& x, const Tensor& weight, float eps, Tensor& output) {
    const int features = weight.size(0);
    const std::size_t rows = x.numel() / features;
    const auto* values = static_cast<const T*>(x.data_ptr());
    const auto* scale = static_cast<const T*>(weight.data_ptr());
    auto* result = static_cast<T*>(output.data_ptr());
    for (std::size_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0f;
        for (int col = 0; col < features; ++col) {
            const float value = static_cast<float>(values[OffsetAt(x, row * features + col)]);
            sum_squares += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_squares / features + eps);
        for (int col = 0; col < features; ++col) {
            const float value = static_cast<float>(values[OffsetAt(x, row * features + col)]);
            const float weight_value = static_cast<float>(scale[col * weight.stride()[0]]);
            result[row * features + col] = static_cast<T>((value * inv_rms) * weight_value);
        }
    }
}

template <typename T>
void BackwardTyped(const Tensor& x, const Tensor& weight, const Tensor& grad,
                   float eps, Tensor& dx, Tensor& dw) {
    const int features = weight.size(0);
    const std::size_t rows = x.numel() / features;
    const auto* values = static_cast<const T*>(x.data_ptr());
    const auto* scale = static_cast<const T*>(weight.data_ptr());
    const auto* upstream = static_cast<const T*>(grad.data_ptr());
    auto* input_grad = static_cast<T*>(dx.data_ptr());
    auto* weight_grad = static_cast<T*>(dw.data_ptr());
    std::vector<float> inv_rms(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0f;
        float sum_grad_weight_x = 0.0f;
        for (int col = 0; col < features; ++col) {
            const float value = static_cast<float>(values[OffsetAt(x, row * features + col)]);
            const float grad_value = static_cast<float>(
                upstream[OffsetAt(grad, row * features + col)]);
            const float weight_value = static_cast<float>(scale[col * weight.stride()[0]]);
            sum_squares += value * value;
            sum_grad_weight_x += grad_value * weight_value * value;
        }
        const float inv = 1.0f / std::sqrt(sum_squares / features + eps);
        inv_rms[row] = inv;
        // 用归一化输入计算梯度，避免先算 RMS 倒数的三次方。
        const float mean_grad_weight_x_hat = (sum_grad_weight_x / features) * inv;
        for (int col = 0; col < features; ++col) {
            const float value = static_cast<float>(values[OffsetAt(x, row * features + col)]);
            const float grad_value = static_cast<float>(
                upstream[OffsetAt(grad, row * features + col)]);
            const float weight_value = static_cast<float>(scale[col * weight.stride()[0]]);
            const float x_hat = value * inv;
            const float weighted_grad = grad_value * weight_value;
            const float dx_value = (weighted_grad - x_hat * mean_grad_weight_x_hat) * inv;
            input_grad[row * features + col] = static_cast<T>(dx_value);
        }
    }
    for (int col = 0; col < features; ++col) {
        float total = 0.0f;
        for (std::size_t row = 0; row < rows; ++row) {
            const float value = static_cast<float>(values[OffsetAt(x, row * features + col)]);
            const float grad_value = static_cast<float>(
                upstream[OffsetAt(grad, row * features + col)]);
            total += grad_value * value * inv_rms[row];
        }
        weight_grad[col] = static_cast<T>(total);
    }
}

}  // namespace

void RMSNormCPU(const Tensor& x, const Tensor& weight, float eps, Tensor& output) {
    DispatchFloatingDtypes(x.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        ForwardTyped<T>(x, weight, eps, output);
    });
}

void RMSNormBackwardCPU(const Tensor& x, const Tensor& weight, const Tensor& grad,
                        float eps, Tensor& dx, Tensor& dw) {
    DispatchFloatingDtypes(x.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        BackwardTyped<T>(x, weight, grad, eps, dx, dw);
    });
}

}  // namespace mtensor
