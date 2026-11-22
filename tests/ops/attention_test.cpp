#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef ATTENTION_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/autograd.h"
#include "mini_infer/ops/attention.h"

namespace {
using namespace mtensor;

#ifdef ATTENTION_TEST_CUDA
constexpr Device kDevice = Device::kCUDA;
#else
constexpr Device kDevice = Device::kCPU;
#endif
constexpr Dtype kFloat = Dtype::kFloat32;
constexpr int kLength = 3;
constexpr int kWidth = 4;
constexpr int kHeads = 2;
constexpr int kSize = kLength * kWidth;
using Values = std::array<float, kSize>;
using InputValues = std::array<Values, 3>;
using ReferenceValues = std::array<double, kSize>;

constexpr InputValues kInputs{{
    {0.2f, 0.8f, -0.3f, 0.2f, 0.4f, 0.1f, 0.5f, 0.1f, 0.6f, -0.5f, 0.7f, -0.2f},
    {0.6f, -0.3f, 0.4f, 0.1f, -0.8f, 0.2f, 0.6f, 0.7f, 0.5f, -0.2f, -0.4f, 0.3f},
    {-0.1f, 0.2f, 0.3f, 0.7f, 0.5f, 0.4f, -0.9f, 0.3f, 0.2f, -0.7f, -0.1f, -0.3f},
}};
constexpr Values kGrad{0.3f, -0.2f, 0.5f, 0.3f, -0.7f, -0.4f,
                       0.2f, 0.5f, 0.5f, 0.8f, -0.2f, 0.3f};

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Tensor MakeTensor(const Values& values, bool strided) {
    const Tensor contiguous(values.data(), {kLength, kWidth}, kFloat, kDevice);
    if (!strided) return contiguous;
    NogradNode guard;
    Tensor view = Tensor({kWidth, kLength}, kFloat, kDevice).transpose(0, 1);
    view.copy_(contiguous);
    Require(!view.is_contiguous(), "attention test must use a non-contiguous view");
    return view;
}

void ExpectNear(const Tensor& actual, const double* expected, std::size_t count,
                const char* message, double tolerance = 2e-5) {
    Require(actual.numel() == count && actual.dtype() == kFloat && actual.device() == kDevice, message);
    NogradNode guard;
    const Tensor host = actual.contiguous().to(Device::kCPU);
    const auto* values = static_cast<const float*>(host.data_ptr());
    for (std::size_t i = 0; i < count; ++i) {
        Require(std::abs(values[i] - expected[i]) <= tolerance, message);
    }
}

// 用独立的 FP64 参考计算，不调用库内 attention 或 softmax。
ReferenceValues ReferenceAttention(const InputValues& inputs) {
    const auto& [q, k, v] = inputs;
    ReferenceValues output{};
    constexpr int head_width = kWidth / kHeads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_width));
    for (int row = 0; row < kLength; ++row) {
        for (int head = 0; head < kHeads; ++head) {
            const int head_start = head * head_width;
            double weights[kLength];
            double partition = 0.0;
            for (int j = 0; j <= row; ++j) {
                double dot = 0.0;
                for (int c = head_start; c < head_start + head_width; ++c) {
                    dot += static_cast<double>(q[row * kWidth + c]) * k[j * kWidth + c];
                }
                // 固定样本的 logits 很小，FP64 参考不需要先减最大值。
                weights[j] = std::exp(dot * scale);
                partition += weights[j];
            }
            for (int j = 0; j <= row; ++j) {
                for (int c = head_start; c < head_start + head_width; ++c) {
                    output[row * kWidth + c] += weights[j] / partition * v[j * kWidth + c];
                }
            }
        }
    }
    return output;
}

double Objective(const InputValues& inputs) {
    const auto output = ReferenceAttention(inputs);
    double result = 0.0;
    for (int i = 0; i < kSize; ++i) result += output[i] * kGrad[i];
    return result;
}

void CheckAttention() {
    std::array<Tensor, 3> inputs;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = MakeTensor(kInputs[i], true);
        inputs[i].requires_grad_(true);
    }

    Tensor output = causal_attention(inputs[0], inputs[1], inputs[2], kHeads);
    Require(output.shape() == std::vector<int>{kLength, kWidth}, "attention output shape");
    const auto node = output.grad_node();
    Require(node != nullptr, "attention must build a backward node");
    const auto expected_output = ReferenceAttention(kInputs);
    ExpectNear(output, expected_output.data(), kSize, "attention forward differs from FP64 reference");
    const Tensor upstream = MakeTensor(kGrad, true);
    std::vector<Tensor> direct_grads;
    {
        NogradNode guard;
        // backward() 会把上游梯度复制成连续布局；直接调用节点覆盖跨步路径。
        direct_grads = node->backward(upstream);
    }
    output.backward(upstream);

    constexpr float epsilon = 1e-3f;
    for (std::size_t which = 0; which < inputs.size(); ++which) {
        ReferenceValues expected{};
        auto perturbed = kInputs;
        for (int i = 0; i < kSize; ++i) {
            const float plus = kInputs[which][i] + epsilon;
            const float minus = kInputs[which][i] - epsilon;
            perturbed[which][i] = plus;
            const double a = Objective(perturbed);
            perturbed[which][i] = minus;
            const double b = Objective(perturbed);
            perturbed[which][i] = kInputs[which][i];
            expected[i] = (a - b) / (static_cast<double>(plus) - minus);
        }
        ExpectNear(inputs[which].grad(), expected.data(), kSize, "attention backward differs from finite difference");
        ExpectNear(direct_grads.at(which), expected.data(), kSize, "attention backward with strided upstream");
    }

    inputs[0].zero_();
    bool rejected = false;
    try { output.backward(upstream); } catch (const std::logic_error&) { rejected = true; }
    Require(rejected, "attention backward must reject input modified after forward");
}

void CheckFutureMask() {
    NogradNode guard;
    auto inputs = kInputs;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int i = (kLength - 1) * kWidth; i < kSize; ++i) {
        inputs[1][i] = nan;
        inputs[2][i] = nan;
    }
    const Tensor output = causal_attention(MakeTensor(inputs[0], false), MakeTensor(inputs[1], false),
                                           MakeTensor(inputs[2], false), kHeads);
    const Tensor prefix = output.narrow(0, 0, kLength - 1);
    const auto expected = ReferenceAttention(kInputs);
    ExpectNear(prefix, expected.data(), (kLength - 1) * kWidth, "attention read a future key or value");
    Require(!output.grad_node(), "attention built a graph under no-grad");
}

}  // namespace

int main() {
    try {
#ifdef ATTENTION_TEST_CUDA
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) return 77;
        Require(status == cudaSuccess, "cudaGetDeviceCount failed");
#endif
        CheckAttention();
        CheckFutureMask();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
