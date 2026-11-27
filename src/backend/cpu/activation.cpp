#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "ops/activation/kernels.h"
#include "mini_infer/tensor/tensor.h"
#include "backend/common/cpu_iteration.h"

namespace mtensor {
namespace {

template <typename Input, typename Output>
void SigmoidTypedCPU(const Tensor& x, Tensor& out) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(out.data_ptr());
    const auto& shape = x.shape();
    const auto& stride = x.stride();
    ForEachOffset(shape, stride,
        [&](std::size_t offset) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const float v = static_cast<float>(value);
            const float t = std::exp(-std::abs(v));
            const Output result(v >= 0.0f ? 1.0f / (1.0f + t) : t / (1.0f + t));
            std::memcpy(dst, &result, sizeof(Output));
            dst += sizeof(Output);
        });
}

template <typename Input, typename Output>
void SiluTypedCPU(const Tensor& x, Tensor& out) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(out.data_ptr());
    const auto& shape = x.shape();
    const auto& stride = x.stride();
    ForEachOffset(shape, stride,
        [&](std::size_t offset) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const float v = static_cast<float>(value);
            float y;
            if (v < -80.0f) {
                // 负尾部用 double，避免 exp(x) 在乘 x 前下溢。
                const double t = std::exp(static_cast<double>(v));
                y = static_cast<float>(v * t / (1.0 + t));
            } else {
                const float t = std::exp(-std::abs(v));
                y = v >= 0.0f ? v / (1.0f + t) : (v * t) / (1.0f + t);
            }
            const Output result(y);
            std::memcpy(dst, &result, sizeof(Output));
            dst += sizeof(Output);
        });
}

template <typename T>
void SigmoidBackwardTypedCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto* input = static_cast<const std::byte*>(x.data_ptr());
    const auto* grad = static_cast<const std::byte*>(output_grad.data_ptr());
    auto* output = static_cast<std::byte*>(dx.data_ptr());
    ForEachOffset(x.shape(), x.stride(), output_grad.stride(),
        [&](std::size_t xi, std::size_t gi) {
            T value(0), upstream(0);
            std::memcpy(&value, input + xi * sizeof(T), sizeof(T));
            std::memcpy(&upstream, grad + gi * sizeof(T), sizeof(T));
            const float t = std::exp(-std::abs(static_cast<float>(value)));
            const float denominator = 1.0f + t;
            // 中间保持 FP32，先乘上游梯度，最后才写回原 dtype。
            const T result((static_cast<float>(upstream) * t) / (denominator * denominator));
            std::memcpy(output, &result, sizeof(T));
            output += sizeof(T);
        });
}

template <typename T>
void SiluBackwardTypedCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto* input = static_cast<const std::byte*>(x.data_ptr());
    const auto* grad = static_cast<const std::byte*>(output_grad.data_ptr());
    auto* output = static_cast<std::byte*>(dx.data_ptr());
    ForEachOffset(x.shape(), x.stride(), output_grad.stride(),
        [&](std::size_t xi, std::size_t gi) {
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
        });
}

}  // namespace

void SigmoidCPU(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: SigmoidTypedCPU<float, float>(x, out); break;
        case Dtype::kFloat16: SigmoidTypedCPU<Float16, Float16>(x, out); break;
        case Dtype::kBFloat16: SigmoidTypedCPU<BFloat16, BFloat16>(x, out); break;
        case Dtype::kInt8: SigmoidTypedCPU<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: SigmoidTypedCPU<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: SigmoidTypedCPU<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: SigmoidTypedCPU<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: SigmoidTypedCPU<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("sigmoid: unsupported dtype");
    }
}

void SiluCPU(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: SiluTypedCPU<float, float>(x, out); break;
        case Dtype::kFloat16: SiluTypedCPU<Float16, Float16>(x, out); break;
        case Dtype::kBFloat16: SiluTypedCPU<BFloat16, BFloat16>(x, out); break;
        case Dtype::kInt8: SiluTypedCPU<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: SiluTypedCPU<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: SiluTypedCPU<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: SiluTypedCPU<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: SiluTypedCPU<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("silu: unsupported dtype");
    }
}

void SigmoidBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: SigmoidBackwardTypedCPU<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: SigmoidBackwardTypedCPU<Float16>(x, output_grad, dx); break;
        case Dtype::kBFloat16: SigmoidBackwardTypedCPU<BFloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("sigmoid backward: unsupported dtype");
    }
}

void SiluBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: SiluBackwardTypedCPU<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: SiluBackwardTypedCPU<Float16>(x, output_grad, dx); break;
        case Dtype::kBFloat16: SiluBackwardTypedCPU<BFloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("silu backward: unsupported dtype");
    }
}

}  // namespace mtensor
