#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mini_infer/ops/norm.h"
#include "mini_infer/autograd.h"
#include "ops/norm/kernels.h"
#include "tensor/check.h"

namespace mtensor {

// FP16/BF16 的 RMSNorm 反向，中间计算使用 float。
class RMSNormNode : public AutogradNode {
public:
    std::uint64_t input_versions[2] = {};
    float eps = 0.0f;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};


Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.dim() == 0) {
        throw std::invalid_argument("rms_norm: x must have a feature dimension");
    }
    if (weight.dim() != 1 || weight.size(0) != x.size(x.dim() - 1)) {
        throw std::invalid_argument("rms_norm: weight must be a vector matching the last dimension of x");
    }
    // y = weight * x / sqrt(mean(x²) + eps)
    if (x.dtype() != Dtype::kFloat16 && x.dtype() != Dtype::kBFloat16) {
        auto inv_rms = ((x * x).mean(-1, true) + eps).rsqrt();
        return (x * inv_rms) * weight;
    }

    CheckSameDtype(x, weight, "rms_norm: x and weight must have the same dtype");
    CheckSameDevice(x, weight, "rms_norm: x and weight must have the same device");
    CheckNotEmpty(x, "rms_norm: x must not be empty");
    CheckNotEmpty(weight, "rms_norm: weight must not be empty");

    Tensor output(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::kCPU) {
        const auto* x_data = static_cast<const std::uint16_t*>(x.data_ptr());
        const auto* weight_data = static_cast<const std::uint16_t*>(weight.data_ptr());
        auto* output_data = static_cast<std::uint16_t*>(output.data_ptr());
        const Dtype dtype = x.dtype();
        const int features = x.size(x.dim() - 1);
        const std::size_t rows = x.numel() / features;
        const std::size_t feature_stride = x.stride().back();

        for (std::size_t row = 0; row < rows; ++row) {
            // 从前导维度拆出坐标，按 stride 找到这一行的起点。
            std::size_t index = row;
            std::size_t row_offset = 0;
            for (int dim = x.dim() - 2; dim >= 0; --dim) {
                row_offset += (index % x.size(dim)) * x.stride()[dim];
                index /= x.size(dim);
            }

            // 平方、累加和均值都保存在 float 中。
            float sum_squares = 0.0f;
            for (int j = 0; j < features; ++j) {
                const auto* element = x_data + row_offset + j * feature_stride;
                float x_value;
                if (dtype == Dtype::kFloat16) {
                    x_value = DecodeFloat16(element);
                } else {
                    x_value = DecodeBFloat16(element);
                }
                sum_squares += x_value * x_value;
            }
            const float mean_square = sum_squares / static_cast<float>(features);
            const float inv_rms = 1.0f / std::sqrt(mean_square + eps);

            // 归一化后乘权重，最后才编码回原 dtype。
            for (int j = 0; j < features; ++j) {
                const auto* x_element = x_data + row_offset + j * feature_stride;
                const auto* weight_element = weight_data + j * weight.stride()[0];
                float x_value, weight_value;
                if (dtype == Dtype::kFloat16) {
                    x_value = DecodeFloat16(x_element);
                    weight_value = DecodeFloat16(weight_element);
                } else {
                    x_value = DecodeBFloat16(x_element);
                    weight_value = DecodeBFloat16(weight_element);
                }
                const float value = (x_value * inv_rms) * weight_value;
                if (dtype == Dtype::kFloat16) {
                    output_data[row * features + j] = EncodeFloat16(value);
                } else {
                    output_data[row * features + j] = EncodeBFloat16(value);
                }
            }
        }
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        RMSNormCUDA(x, weight, eps, output);
#endif
    } else {
        throw std::logic_error("rms_norm: no backend for this device");
    }

    if (EnableGrad()) {
        auto node = std::make_shared<RMSNormNode>();
        node->inputs = {x, weight};
        node->input_versions[0] = x.version();
        node->input_versions[1] = weight.version();
        node->eps = eps;
        output.set_grad_node(node);
    }
    return output;
}


std::vector<Tensor> RMSNormNode::backward(const Tensor& output_grad) {
    const Tensor& x = inputs[0];
    const Tensor& weight = inputs[1];
    CheckVersion(x, input_versions[0], "RMSNormNode::backward: input modified after forward");
    CheckVersion(weight, input_versions[1], "RMSNormNode::backward: weight modified after forward");

    Tensor dx(x.shape(), x.dtype(), x.device());
    Tensor dw(weight.shape(), weight.dtype(), weight.device());
#ifdef ENABLE_CUDA
    if (x.device() == Device::kCUDA) {
        RMSNormBackwardCUDA(x, weight, output_grad, eps, dx, dw);
        return {dx, dw};
    }
#endif
    const auto* x_data = static_cast<const std::uint16_t*>(x.data_ptr());
    const auto* weight_data = static_cast<const std::uint16_t*>(weight.data_ptr());
    const auto* grad_data = static_cast<const std::uint16_t*>(output_grad.data_ptr());
    auto* dx_data = static_cast<std::uint16_t*>(dx.data_ptr());
    auto* dw_data = static_cast<std::uint16_t*>(dw.data_ptr());
    const Dtype dtype = x.dtype();
    const int features = x.size(x.dim() - 1);
    const std::size_t rows = x.numel() / features;
    const std::size_t x_stride = x.stride().back();
    const std::size_t grad_stride = output_grad.stride().back();
    const std::size_t weight_stride = weight.stride()[0];
    std::vector<float> weight_grad(features, 0.0f);

    for (std::size_t row = 0; row < rows; ++row) {
        // 输入和上游梯度可以有不同的 stride，分别计算这一行的起点。
        std::size_t index = row;
        std::size_t x_offset = 0;
        std::size_t grad_offset = 0;
        for (int dim = x.dim() - 2; dim >= 0; --dim) {
            x_offset += (index % x.size(dim)) * x.stride()[dim];
            grad_offset += (index % x.size(dim)) * output_grad.stride()[dim];
            index /= x.size(dim);
        }
        const auto* x_row = x_data + x_offset;
        const auto* grad_row = grad_data + grad_offset;
        auto* dx_row = dx_data + row * features;

        float sum_squares = 0.0f;
        float sum_grad_weight_x = 0.0f;
        for (int j = 0; j < features; ++j) {
            float x_value, grad_value, weight_value;
            if (dtype == Dtype::kFloat16) {
                x_value = DecodeFloat16(x_row + j * x_stride);
                grad_value = DecodeFloat16(grad_row + j * grad_stride);
                weight_value = DecodeFloat16(weight_data + j * weight_stride);
            } else {
                x_value = DecodeBFloat16(x_row + j * x_stride);
                grad_value = DecodeBFloat16(grad_row + j * grad_stride);
                weight_value = DecodeBFloat16(weight_data + j * weight_stride);
            }
            sum_squares += x_value * x_value;
            sum_grad_weight_x += grad_value * weight_value * x_value;
        }
        const float mean_square = sum_squares / static_cast<float>(features);
        const float inv_rms = 1.0f / std::sqrt(mean_square + eps);
        const float correction = inv_rms * inv_rms * inv_rms *
            (sum_grad_weight_x / static_cast<float>(features));

        for (int j = 0; j < features; ++j) {
            float x_value, grad_value, weight_value;
            if (dtype == Dtype::kFloat16) {
                x_value = DecodeFloat16(x_row + j * x_stride);
                grad_value = DecodeFloat16(grad_row + j * grad_stride);
                weight_value = DecodeFloat16(weight_data + j * weight_stride);
            } else {
                x_value = DecodeBFloat16(x_row + j * x_stride);
                grad_value = DecodeBFloat16(grad_row + j * grad_stride);
                weight_value = DecodeBFloat16(weight_data + j * weight_stride);
            }
            // dx = g * weight * r - x * r³ * mean(g * weight * x)
            const float dx_value = grad_value * weight_value * inv_rms - x_value * correction;
            if (dtype == Dtype::kFloat16) {
                dx_row[j] = EncodeFloat16(dx_value);
            } else {
                dx_row[j] = EncodeBFloat16(dx_value);
            }
            weight_grad[j] += grad_value * x_value * inv_rms;
        }
    }

    // 权重梯度跨所有行累加完成后，再写回原 dtype。
    for (int j = 0; j < features; ++j) {
        if (dtype == Dtype::kFloat16) {
            dw_data[j] = EncodeFloat16(weight_grad[j]);
        } else {
            dw_data[j] = EncodeBFloat16(weight_grad[j]);
        }
    }
    return {dx, dw};
}

}  // namespace mtensor
