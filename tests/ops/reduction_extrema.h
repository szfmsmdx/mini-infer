#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mini_infer/autograd.h"
#include "mini_infer/ops/reduction.h"

namespace mtensor::test {

inline void RequireExtrema(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void CheckExtremaDtype(Device device, Dtype dtype, T low, T high) {
    NogradNode guard;
    const T values[] = {low, high};
    const Tensor input(values, {2}, dtype, device);
    const Tensor minimum = input.min().to(Device::kCPU);
    const Tensor maximum = input.max().to(Device::kCPU);
    RequireExtrema(minimum.is_scalar() && maximum.is_scalar() &&
                       minimum.dtype() == dtype && maximum.dtype() == dtype,
                   "max/min must preserve dtype and return scalars");
    RequireExtrema(*static_cast<const T*>(minimum.data_ptr()) == low &&
                       *static_cast<const T*>(maximum.data_ptr()) == high,
                   "max/min dtype values");
}

inline void CheckExtremaAxes(Device device) {
    const float values[] = {2, 5, 5, -4, 1, -4};
    Tensor input(values, {2, 3}, Dtype::kFloat32, device);
    input.requires_grad_(true);
    const Tensor view = input.transpose(0, 1);
    Tensor maxima = mtensor::max(view, 0, true);
    const Tensor minima = input.min(-1);
    {
        NogradNode guard;
        const Tensor hi = maxima.to(Device::kCPU);
        const auto* hi_data = static_cast<const float*>(hi.data_ptr());
        RequireExtrema(hi.shape() == std::vector<int>{1, 2} && hi_data[0] == 5 && hi_data[1] == 1,
                       "strided max keepdim");
        const Tensor lo = minima.to(Device::kCPU);
        const auto* lo_data = static_cast<const float*>(lo.data_ptr());
        RequireExtrema(lo.shape() == std::vector<int>{2} && lo_data[0] == 2 && lo_data[1] == -4,
                       "min along negative axis");
    }

    const float weights[] = {2, 4};
    maxima.backward(Tensor(weights, {1, 2}, Dtype::kFloat32, device));
    {
        NogradNode guard;
        const Tensor grad = input.grad().to(Device::kCPU);
        const float expected[] = {0, 1, 1, 0, 4, 0};
        const auto* data = static_cast<const float*>(grad.data_ptr());
        for (std::size_t i = 0; i < grad.numel(); ++i) {
            RequireExtrema(data[i] == expected[i], "strided axis max gradient");
        }
    }

    input.zero_grad();
    mtensor::min(view).backward();
    {
        NogradNode guard;
        const Tensor grad = input.grad().to(Device::kCPU);
        const float expected[] = {0, 0, 0, 0.5f, 0, 0.5f};
        const auto* data = static_cast<const float*>(grad.data_ptr());
        for (std::size_t i = 0; i < grad.numel(); ++i) {
            RequireExtrema(data[i] == expected[i], "global min must split ties");
        }
    }

    Tensor changed = input.max();
    changed.fill_(0.0f);
    bool rejected = false;
    try { changed.backward(); } catch (const std::logic_error&) { rejected = true; }
    RequireExtrema(rejected, "max backward must reject a changed saved output");
    rejected = false;
    try { view.min(2); } catch (const std::out_of_range&) { rejected = true; }
    RequireExtrema(rejected, "min must reject an invalid axis");
    rejected = false;
    try { mtensor::max(Tensor{}); } catch (const std::invalid_argument&) { rejected = true; }
    RequireExtrema(rejected, "max must reject empty input");
}

inline void CheckExtremaHalfGrad(Device device, Dtype dtype) {
    Tensor input = full({3}, 2.0f, dtype, device);
    input.requires_grad_(true);
    input.max().backward(full({}, 3.0f, dtype, device));
    NogradNode guard;
    const Tensor grad = input.grad().to(Device::kCPU);
    const auto* bits = static_cast<const std::uint16_t*>(grad.data_ptr());
    const auto one = dtype == Dtype::kFloat16 ? EncodeFloat16(1.0f) : EncodeBFloat16(1.0f);
    RequireExtrema(grad.dtype() == dtype && grad.shape() == std::vector<int>{3} &&
                       bits[0] == one && bits[1] == one && bits[2] == one,
                   "half max must split gradients and preserve dtype");
}

inline void CheckExtremaSpecialValues(Device device) {
    NogradNode guard;
    std::vector<float> values(257, 0.0f);
    values.front() = -0.0f;
    values.back() = 7.0f;
    const Tensor input(values.data(), {257}, Dtype::kFloat32, device);
    const Tensor minimum = input.min().to(Device::kCPU);
    const Tensor maximum = input.max().to(Device::kCPU);
    RequireExtrema(std::signbit(*static_cast<const float*>(minimum.data_ptr())) &&
                       *static_cast<const float*>(maximum.data_ptr()) == 7.0f,
                   "max/min with signed zeros and a 257-element reduction");
    values.back() = std::numeric_limits<float>::quiet_NaN();
    const Tensor nan_input(values.data(), {257}, Dtype::kFloat32, device);
    const Tensor nan_min = nan_input.min().to(Device::kCPU);
    const Tensor nan_max = nan_input.max().to(Device::kCPU);
    RequireExtrema(std::isnan(*static_cast<const float*>(nan_min.data_ptr())) &&
                       std::isnan(*static_cast<const float*>(nan_max.data_ptr())),
                   "max/min must propagate NaN");
}

inline void CheckExtrema(Device device) {
    CheckExtremaDtype<std::int8_t>(device, Dtype::kInt8, -9, 2);
    CheckExtremaDtype<std::uint8_t>(device, Dtype::kUInt8, 2, 250);
    CheckExtremaDtype<std::int16_t>(device, Dtype::kInt16, -30000, 29);
    CheckExtremaDtype<std::int32_t>(device, Dtype::kInt32, -2000000000, 11);
    constexpr auto high = std::numeric_limits<std::int64_t>::max();
    CheckExtremaDtype<std::int64_t>(device, Dtype::kInt64, high - 1, high);
    CheckExtremaDtype<float>(device, Dtype::kFloat32, -2.0f, 3.0f);
    CheckExtremaDtype<std::uint16_t>(device, Dtype::kFloat16,
                                    EncodeFloat16(-2.0f), EncodeFloat16(3.0f));
    CheckExtremaDtype<std::uint16_t>(device, Dtype::kBFloat16,
                                    EncodeBFloat16(-2.0f), EncodeBFloat16(3.0f));
    CheckExtremaAxes(device);
    CheckExtremaHalfGrad(device, Dtype::kFloat16);
    CheckExtremaHalfGrad(device, Dtype::kBFloat16);
    CheckExtremaSpecialValues(device);
}

}  // namespace mtensor::test
