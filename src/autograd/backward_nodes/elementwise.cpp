#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "autograd/backward_nodes/elementwise.h"
#include "mini_infer/ops/elementwise.h"
#include "mini_infer/ops/activation.h"
#include "tensor/check.h"

namespace mtensor {
namespace {

template <typename T>
void SigmoidBackwardTypedCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto* input = static_cast<const std::byte*>(x.data_ptr());
    const auto* grad = static_cast<const std::byte*>(output_grad.data_ptr());
    auto* output = static_cast<std::byte*>(dx.data_ptr());
    auto visit = [&](auto&& self, int dim, std::size_t xi, std::size_t gi) -> void {
        if (dim == x.dim()) {
            T value(0), upstream(0);
            std::memcpy(&value, input + xi * sizeof(T), sizeof(T));
            std::memcpy(&upstream, grad + gi * sizeof(T), sizeof(T));
            const float t = std::exp(-std::abs(static_cast<float>(value)));
            const float denominator = 1.0f + t;
            // 中间保持 FP32，先乘上游梯度，最后才写回原 dtype。
            const T result((static_cast<float>(upstream) * t) / (denominator * denominator));
            std::memcpy(output, &result, sizeof(T));
            output += sizeof(T);
            return;
        }
        for (int i = 0; i < x.size(dim); ++i) {
            self(self, dim + 1, xi + i * x.stride()[dim],
                 gi + i * output_grad.stride()[dim]);
        }
    };
    visit(visit, 0, 0, 0);
}

void SigmoidBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: SigmoidBackwardTypedCPU<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: SigmoidBackwardTypedCPU<Float16>(x, output_grad, dx); break;
        case Dtype::kBFloat16: SigmoidBackwardTypedCPU<BFloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("sigmoid backward: unsupported dtype");
    }
}

template <typename T>
void RsqrtBackwardTypedCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto* input = static_cast<const std::byte*>(x.data_ptr());
    const auto* grad = static_cast<const std::byte*>(output_grad.data_ptr());
    auto* output = static_cast<std::byte*>(dx.data_ptr());
    auto visit = [&](auto&& self, int dim, std::size_t xi, std::size_t gi) -> void {
        if (dim == x.dim()) {
            T value(0), upstream(0);
            std::memcpy(&value, input + xi * sizeof(T), sizeof(T));
            std::memcpy(&upstream, grad + gi * sizeof(T), sizeof(T));
            const float y = 1.0f / std::sqrt(static_cast<float>(value));
            // y 和乘法中间结果保持 FP32，最后才写回原 dtype。
            const T result(((-0.5f * y) * static_cast<float>(upstream)) * y * y);
            std::memcpy(output, &result, sizeof(T));
            output += sizeof(T);
            return;
        }
        for (int i = 0; i < x.size(dim); ++i) {
            self(self, dim + 1, xi + i * x.stride()[dim],
                 gi + i * output_grad.stride()[dim]);
        }
    };
    visit(visit, 0, 0, 0);
}

void RsqrtBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: RsqrtBackwardTypedCPU<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: RsqrtBackwardTypedCPU<Float16>(x, output_grad, dx); break;
        case Dtype::kBFloat16: RsqrtBackwardTypedCPU<BFloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("rsqrt backward: unsupported dtype");
    }
}

template <typename T>
void SiluBackwardTypedCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto* input = static_cast<const std::byte*>(x.data_ptr());
    const auto* grad = static_cast<const std::byte*>(output_grad.data_ptr());
    auto* output = static_cast<std::byte*>(dx.data_ptr());
    auto visit = [&](auto&& self, int dim, std::size_t xi, std::size_t gi) -> void {
        if (dim == x.dim()) {
            T value(0), upstream(0);
            std::memcpy(&value, input + xi * sizeof(T), sizeof(T));
            std::memcpy(&upstream, grad + gi * sizeof(T), sizeof(T));
            const float x_value = static_cast<float>(value);
            const float g = static_cast<float>(upstream);
            const float s = 1.0f / (1.0f + std::exp(-x_value));
            // 与 Torch 相同的运算顺序，中间保持 FP32，最后写回原 dtype。
            const T result((g * s) * (1.0f + x_value * (1.0f - s)));
            std::memcpy(output, &result, sizeof(T));
            output += sizeof(T);
            return;
        }
        for (int i = 0; i < x.size(dim); ++i) {
            self(self, dim + 1, xi + i * x.stride()[dim],
                 gi + i * output_grad.stride()[dim]);
        }
    };
    visit(visit, 0, 0, 0);
}

void SiluBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: SiluBackwardTypedCPU<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: SiluBackwardTypedCPU<Float16>(x, output_grad, dx); break;
        case Dtype::kBFloat16: SiluBackwardTypedCPU<BFloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("silu backward: unsupported dtype");
    }
}

}  // namespace

std::vector<Tensor> AddNode::backward(const Tensor& output_grad) {
    return {output_grad, output_grad};
}

std::vector<Tensor> SubNode::backward(const Tensor& output_grad) {
    return {output_grad, neg(output_grad)};
}

std::vector<Tensor> NegNode::backward(const Tensor& output_grad) {
    return {neg(output_grad)};
}

std::vector<Tensor> SqrtNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_version, "SqrtNode::backward: input modified after forward");
    const Tensor root = sqrt(inputs[0]);
    return {output_grad / (root + root)};
}

std::vector<Tensor> RsqrtNode::backward(const Tensor& output_grad) {
    const Tensor& x = inputs[0];
    CheckVersion(x, input_version, "RsqrtNode::backward: input modified after forward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::kCPU) {
        RsqrtBackwardCPU(x, output_grad, dx);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        RsqrtBackwardCUDA(x, output_grad, dx);
#endif
    } else {
        throw std::logic_error("rsqrt backward: no backend for this device");
    }
    return {dx};
}

std::vector<Tensor> ExpNode::backward(const Tensor& output_grad) {
    CheckVersion(y, output_version, "ExpNode::backward: output modified after forward");
    return {output_grad * y};
}

std::vector<Tensor> SigmoidNode::backward(const Tensor& output_grad) {
    const Tensor& x = inputs[0];
    CheckVersion(x, input_version, "SigmoidNode::backward: input modified after forward");
    CheckVersion(y, output_version, "SigmoidNode::backward: output modified after forward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::kCPU) {
        SigmoidBackwardCPU(x, output_grad, dx);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        SigmoidBackwardCUDA(x, output_grad, dx);
#endif
    } else {
        throw std::logic_error("sigmoid backward: no backend for this device");
    }
    return {dx};
}

std::vector<Tensor> SiluNode::backward(const Tensor& output_grad) {
    const Tensor& x = inputs[0];
    CheckVersion(x, input_version, "SiluNode::backward: input modified after forward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::kCPU) {
        SiluBackwardCPU(x, output_grad, dx);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        SiluBackwardCUDA(x, output_grad, dx);
#endif
    } else {
        throw std::logic_error("silu backward: no backend for this device");
    }
    return {dx};
}

std::vector<Tensor> MulNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_versions[0], "MulNode::backward: input modified after forward");
    CheckVersion(inputs[1], input_versions[1], "MulNode::backward: input modified after forward");
    return {mul(output_grad, inputs[1]), mul(output_grad, inputs[0])};
}

std::vector<Tensor> DivNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_versions[0], "DivNode::backward: input modified after forward");
    CheckVersion(inputs[1], input_versions[1], "DivNode::backward: input modified after forward");
    return {div(output_grad, inputs[1]),
            -output_grad * ((inputs[0] / inputs[1]) / inputs[1])};
}

}  // namespace mtensor
