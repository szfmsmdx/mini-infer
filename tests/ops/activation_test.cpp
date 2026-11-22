#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef ACTIVATION_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/ops/activation.h"
#include "mini_infer/autograd.h"

namespace {
using namespace mtensor;

template <typename T>
void Near(const Tensor& tensor, const float* expected, float rel_tol, float abs_tol = 0) {
    const Tensor host = tensor.contiguous().to(Device::kCPU);
    const auto* p = static_cast<const T*>(host.data_ptr());
    for (std::size_t i = 0; i < host.numel(); ++i) {
        if (!(std::abs(static_cast<float>(p[i]) - expected[i]) <= abs_tol + rel_tol * std::abs(expected[i]))) {
            std::cerr << "activation mismatch at " << i << ": actual="
                      << static_cast<float>(p[i]) << " expected=" << expected[i]
                      << " rtol=" << rel_tol << " atol=" << abs_tol << std::endl;
            throw std::runtime_error("unexpected activation value");
        }
    }
}

template <typename T>
void CheckFloating(Device device, Dtype dtype, float tol) {
    const T values[] = {T(-12), T(1), T(0), T(-1)};
    Tensor x(values, {2, 2}, dtype, device);
    x.requires_grad_(true);
    const Tensor v = x.transpose(0, 1);
    Tensor y = sigmoid(v);
    const float s[] = {6.144175e-6f, 0.5f, 0.7310586f, 0.2689414f};
    if (y.shape() != v.shape() || y.device() != device || y.dtype() != dtype)
        throw std::runtime_error("incorrect sigmoid metadata");
    Near<T>(y, s, tol);
    const float f[] = {-12 * s[0], 0, s[2], -s[3]};
    Tensor z = silu(v);
    if (z.shape() != v.shape() || z.device() != device || z.dtype() != dtype || !z.grad_node())
        throw std::runtime_error("incorrect silu metadata or missing backward node");
    Near<T>(z, f, tol);
    const T grads[] = {T(2), T(1), T(3), T(4)};
    const Tensor g(grads, {2, 2}, dtype, device);
    z.backward(g);
    const float dz[] = {2*s[0]*(1-12*(1-s[0])), 3*s[2]*(2-s[2]), 0.5f, 4*s[3]*s[3]};
    Near<T>(x.grad(), dz, tol);
    x.zero_grad();
    y.backward(g);
    const float dx[] = {2*s[0]*(1-s[0]), 3*s[2]*(1-s[2]), 0.25f, 4*s[3]*(1-s[3])};
    Near<T>(x.grad(), dx, tol);
    Tensor input_check = sigmoid(v);
    Tensor silu_input_check = silu(v);
    x.zero_();
    bool input_rejected = false;
    try { input_check.backward(g); } catch (const std::logic_error&) { input_rejected = true; }
    if (!input_rejected) throw std::runtime_error("sigmoid accepted a modified input");
    bool silu_input_rejected = false;
    try { silu_input_check.backward(g); } catch (const std::logic_error&) { silu_input_rejected = true; }
    if (!silu_input_rejected) throw std::runtime_error("silu accepted a modified input");

    y = sigmoid(v);
    y.zero_();
    try { y.backward(g); } catch (const std::logic_error&) { return; }
    throw std::runtime_error("sigmoid accepted a modified output");
}

template <typename T>
void CheckSiluForward(Device device, Dtype dtype, double rtol) {
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> samples = {
        -200, -104, -100, -90, std::nextafter(-80.0f, -inf), -80,
        std::nextafter(-80.0f, inf), -20, -17, -1, -0.0f, 0, 1, 20, 200,
        -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        -inf, inf, std::numeric_limits<float>::quiet_NaN()};
    std::vector<T> values;
    for (float v : samples) values.emplace_back(v);
    const Tensor x(values.data(), {static_cast<int>(values.size())}, dtype, device);
    NogradNode no_grad;
    const Tensor y = silu(x);
    if (y.grad_node()) throw std::runtime_error("silu built a graph under no-grad");
    const Tensor host = y.to(Device::kCPU);
    const auto* actual = static_cast<const T*>(host.data_ptr());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double v = static_cast<float>(values[i]);
        const double t = std::exp(-std::abs(v));
        const double exact = v >= 0 ? v/(1+t) : v*t/(1+t);
        const float expected = static_cast<float>(T(static_cast<float>(exact)));
        const float got = static_cast<float>(actual[i]);
        if (std::isnan(expected)) {
            if (!std::isnan(got)) throw std::runtime_error("silu NaN mismatch");
        } else if (std::isinf(expected)) {
            if (got != expected) throw std::runtime_error("silu infinity mismatch");
        } else if (!std::isfinite(got) || ((got == 0) != (expected == 0)) ||
                   std::abs(got-exact) > 1e-5 + rtol*std::abs(exact)) {
            throw std::runtime_error("silu forward tail or tolerance mismatch");
        }
        if (v == 0 && std::signbit(got) != std::signbit(v))
            throw std::runtime_error("silu zero sign mismatch");
    }
}

void CheckSiluBackwardLargeGradient(Device device) {
    const float largest = std::numeric_limits<float>::max();
    const float values[] = {-200, -200, 200, 200};
    const float gradients[] = {largest, -largest, largest, -largest};
    const float expected[] = {0, 0, largest, -largest};
    Tensor x(values, {4}, Dtype::kFloat32, device);
    x.requires_grad_(true);
    Tensor y = silu(x);
    y.backward(Tensor(gradients, {4}, Dtype::kFloat32, device));
    Near<float>(x.grad(), expected, 0);
}

template <typename T>
void CheckSigmoidBackwardPrecision(Device device, Dtype dtype, float largest, float rtol) {
    // 包含 BF16 的四个失败样本、正负尾部和正向饱和输入，跨过两个 CUDA block。
    constexpr int count = 513;
    const float inputs[] = {-92, -92, -90.5f, -90.5f, 12, -12, 0, 8};
    const float gradients[] = {largest, -largest, largest, -largest, largest, -largest, 1, 1};
    std::vector<T> values, upstream;
    std::vector<float> expected;
    for (int i = 0; i < count; ++i) {
        values.emplace_back(inputs[i % 8]);
        upstream.emplace_back(gradients[i % 8]);
        const double t = std::exp(-std::abs(static_cast<double>(static_cast<float>(values.back()))));
        const double g = static_cast<float>(upstream.back());
        expected.push_back(static_cast<float>(T(static_cast<float>((g * t) / ((1 + t) * (1 + t))))));
    }
    Tensor x_storage(std::vector<int>{count, 2}, dtype, device);
    Tensor g_storage(std::vector<int>{count, 3}, dtype, device);
    Tensor x = x_storage.transpose(0, 1)[1];
    Tensor g = g_storage.transpose(0, 1)[2];
    x.copy_(Tensor(values.data(), {count}, dtype, device));
    g.copy_(Tensor(upstream.data(), {count}, dtype, device));
    x.requires_grad_(true);
    Tensor y = sigmoid(x);
    {
        // 直接验证非连续上游梯度；引擎入口会先 clone 梯度。
        NogradNode no_grad;
        const auto gradients = y.grad_node()->backward(g);
        if (gradients.size() != 1 || gradients[0].shape() != x.shape() ||
            gradients[0].dtype() != dtype || gradients[0].device() != device)
            throw std::runtime_error("incorrect sigmoid gradient metadata");
        Near<T>(gradients[0], expected.data(), rtol, 1e-5f);
    }
    y.backward(g);
    Near<T>(x.grad(), expected.data(), rtol, 1e-5f);

    const T zero(0), one(1);
    Tensor scalar(&zero, {}, dtype, device);
    scalar.requires_grad_(true);
    sigmoid(scalar).backward(Tensor(&one, {}, dtype, device));
    const float scalar_expected = 0.25f;
    Near<T>(scalar.grad(), &scalar_expected, 0);
}

template <typename T>
void CheckSiluBackwardPrecision(Device device, Dtype dtype, float rtol) {
    constexpr int count = 513;
    const float inputs[] = {-1.267578125f, -2.84375f, -1.1953125f, -1.34375f, 0, 12};
    const float grads[] = {1, -0.5f, 1, -0.5f, 2, 3};
    std::vector<T> values, upstream;
    std::vector<float> expected;
    for (int i = 0; i < count; ++i) {
        values.emplace_back(inputs[i % 6]);
        upstream.emplace_back(grads[i % 6]);
        const double x = static_cast<float>(values.back());
        const double g = static_cast<float>(upstream.back());
        const double t = std::exp(-std::abs(x));
        const double s = x >= 0 ? 1 / (1 + t) : t / (1 + t);
        expected.push_back(static_cast<float>(T(static_cast<float>(g * (s + x*t/((1+t)*(1+t)))))));
    }
    Tensor xs(std::vector<int>{count, 2}, dtype, device);
    Tensor gs(std::vector<int>{count, 3}, dtype, device);
    Tensor x = xs.transpose(0, 1)[1];
    Tensor g = gs.transpose(0, 1)[2];
    x.copy_(Tensor(values.data(), {count}, dtype, device));
    g.copy_(Tensor(upstream.data(), {count}, dtype, device));
    x.requires_grad_(true);
    Tensor y = silu(x);
    {
        NogradNode no_grad;
        const auto dx = y.grad_node()->backward(g);
        if (dx.size() != 1 || dx[0].shape() != x.shape() ||
            dx[0].dtype() != dtype || dx[0].device() != device)
            throw std::runtime_error("incorrect silu gradient metadata");
        Near<T>(dx[0], expected.data(), rtol, 1e-5f);
    }
    y.backward(g);
    Near<T>(x.grad(), expected.data(), rtol, 1e-5f);
    const T zero(0), one(1);
    Tensor scalar(&zero, {}, dtype, device);
    scalar.requires_grad_(true);
    silu(scalar).backward(Tensor(&one, {}, dtype, device));
    const float scalar_expected = 0.5f;
    Near<T>(scalar.grad(), &scalar_expected, 0);
}

template <typename T>
void CheckIntegerSigmoid(Device device, Dtype dtype, T value) {
    Tensor x(&value, {}, dtype, device);
    const float v = static_cast<float>(value);
    const float s = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(v))));
    Tensor y = sigmoid(x);
    if (y.dtype() != Dtype::kFloat32 || y.grad_node())
        throw std::runtime_error("integer sigmoid must return FP32 without a grad node");
    Near<float>(y, &s, 1e-6f);
    const Tensor z = silu(x);
    if (z.dtype() != Dtype::kFloat32 || z.grad_node())
        throw std::runtime_error("integer silu must return FP32 without a grad node");
    const float expected = v * s;
    Near<float>(z, &expected, 1e-6f);
}
}  // namespace

int main() {
    using namespace mtensor;
#ifdef ACTIVATION_TEST_CUDA
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
        (status == cudaSuccess && count == 0)) return 77;
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    constexpr auto device = Device::kCUDA;
#else
    constexpr auto device = Device::kCPU;
#endif
    CheckFloating<float>(device, Dtype::kFloat32, 1e-6f);
    CheckFloating<Float16>(device, Dtype::kFloat16, 0.01f);
    CheckFloating<BFloat16>(device, Dtype::kBFloat16, 0.02f);
    CheckSiluForward<float>(device, Dtype::kFloat32, 1.3e-6);
    CheckSiluForward<Float16>(device, Dtype::kFloat16, 1e-3);
    CheckSiluForward<BFloat16>(device, Dtype::kBFloat16, 1.6e-2);
    CheckSiluBackwardLargeGradient(device);
    CheckSiluBackwardPrecision<float>(device, Dtype::kFloat32, 1.3e-6f);
    CheckSiluBackwardPrecision<Float16>(device, Dtype::kFloat16, 1e-3f);
    CheckSiluBackwardPrecision<BFloat16>(device, Dtype::kBFloat16, 1.6e-2f);
    CheckSigmoidBackwardPrecision<float>(device, Dtype::kFloat32,
                                        std::numeric_limits<float>::max(), 1.3e-6f);
    CheckSigmoidBackwardPrecision<Float16>(device, Dtype::kFloat16, 65504.0f, 1e-3f);
    CheckSigmoidBackwardPrecision<BFloat16>(device, Dtype::kBFloat16,
                                           3.3895313892515355e38f, 1.6e-2f);
    CheckIntegerSigmoid<std::uint8_t>(device, Dtype::kUInt8, 1);
    CheckIntegerSigmoid<std::int8_t>(device, Dtype::kInt8, -12);
    CheckIntegerSigmoid<std::int16_t>(device, Dtype::kInt16, -20);
    CheckIntegerSigmoid<std::int32_t>(device, Dtype::kInt32, 20);
    CheckIntegerSigmoid<std::int64_t>(device, Dtype::kInt64, 2);
}
