#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef TRANSFORMER_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/autograd.h"
#include "mini_infer/models/layers/simple_decoder_layer.h"
#include "mini_infer/modules/embedding.h"
#include "mini_infer/ops/transformer.h"

namespace {
using namespace mtensor;
#ifdef TRANSFORMER_TEST_CUDA
constexpr Device kDevice = Device::kCUDA;
#else
constexpr Device kDevice = Device::kCPU;
#endif
constexpr Dtype kFloat = Dtype::kFloat32;

void Expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> Values(const Tensor& x) {
    NogradNode guard;
    const Tensor host = x.contiguous().to(Device::kCPU);
    const auto* values = static_cast<const float*>(host.data_ptr());
    return {values, values + host.numel()};
}

void ExpectNear(const Tensor& result, const std::vector<float>& expected, const char* message,
                float tolerance = 2e-5f) {
    const std::vector<float> actual = Values(result);
    Expect(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        Expect(std::abs(actual[i] - expected[i]) <= tolerance, message);
    }
}

template <class Error, class Action>
void ExpectError(Action action, const char* message) {
    try {
        action();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

void CheckEmbedding() {
    const float weight_data[] = {1, 2, 3, 4, 5, 6};
    Tensor source(weight_data, {2, 3}, kFloat, kDevice);
    source.requires_grad_(true);
    const Tensor weight = source.transpose(0, 1);
    const std::int32_t index_data[] = {2, -1, 0, -1, 2, -1};
    Tensor index_storage(index_data, {3, 2}, Dtype::kInt32, kDevice);
    const Tensor indices = index_storage.narrow(1, 0, 1).squeeze(1);
    Tensor result = embedding(weight, indices);
    Expect(result.shape() == std::vector<int>({3, 2}), "embedding output shape");
    ExpectNear(result, {3, 6, 1, 4, 3, 6}, "embedding gather with strides");
    result.sum().backward();
    ExpectNear(source.grad(), {1, 0, 2, 1, 0, 2}, "embedding scatter-add for repeated tokens");

    index_storage.fill_(0);
    ExpectError<std::logic_error>([&] { result.sum().backward(); }, "embedding index version");
    for (const std::int32_t invalid : {-1, 3}) {
        const std::int32_t mixed[] = {1, invalid};
        const Tensor bad_indices(mixed, {2}, Dtype::kInt32, kDevice);
        ExpectError<std::out_of_range>([&] { embedding(weight, bad_indices); }, "embedding bounds");
    }
}

void CheckRoPE() {
    const float values[] = {1, 1, 0, 2, 1, 1, 0, -1, 1, 1, 0, 3, 1, 1, 0, 4};
    Tensor source(values, {8, 2}, kFloat, kDevice);
    source.requires_grad_(true);
    const Tensor input = source.transpose(0, 1);
    Tensor result = rope(input, 2);
    const float c = std::cos(1.0f);
    const float s = std::sin(1.0f);
    const float c2 = std::cos(0.01f);
    const float s2 = std::sin(0.01f);
    ExpectNear(result, {1, 0, 1, 0, 1, 0, 1, 0,
                        c - 2 * s, s + 2 * c, c2 + s2, s2 - c2,
                        c - 3 * s, s + 3 * c, c2 - 4 * s2, s2 + 4 * c2},
               "rope rotation, frequency and head reset");
    result.backward(input.detach());
    ExpectNear(source.grad().transpose(0, 1), {1, 0, 1, 0, 1, 0, 1, 0,
                                             c + 2 * s, 2 * c - s, c2 - s2, -c2 - s2,
                                             c + 3 * s, 3 * c - s, c2 + 4 * s2, 4 * c2 - s2},
               "rope inverse rotation in backward");
}

void CheckEmbeddingLayer() {
    Embedding layer(3, 2, kDevice);
    auto parameters = layer.parameters();
    Expect(parameters.size() == 1 && parameters[0].shape() == std::vector<int>{3, 2} &&
               parameters[0].requires_grad(),
           "Embedding must register its trainable weight");
    const float weights[] = {1, 2, 3, 4, 5, 6};
    parameters[0].copy_(Tensor(weights, {3, 2}, kFloat, kDevice));
    const std::int32_t ids[] = {2, 0, 2};
    const Tensor indices(ids, {3}, Dtype::kInt32, kDevice);
    Tensor output = layer(indices);
    ExpectNear(output, {5, 6, 1, 2, 5, 6}, "Embedding lookup uses the registered weight");
    output.sum().backward();
    ExpectNear(parameters[0].grad(), {1, 1, 0, 0, 2, 2}, "Embedding registered weight gradient");
}

void CheckDecoderDimensions() {
    constexpr float eps = 1e-5f;
    ExpectError<std::invalid_argument>([=] { SimpleDecoderLayer layer(12, 4, eps, kDevice); },
                                       "decoder must reject an odd head width");
    ExpectError<std::invalid_argument>([=] { SimpleDecoderLayer layer(8, 0, eps, kDevice); },
                                       "decoder must reject zero heads");
    constexpr int too_wide = std::numeric_limits<int>::max() / 4 + 1;
    ExpectError<std::invalid_argument>([=] { SimpleDecoderLayer layer(too_wide, 1, eps, kDevice); },
                                       "decoder must reject a 4C overflow before creating sublayers");
}

void CheckDecoderForwardBackward() {
    constexpr float eps = 1.0f / 128;
    SimpleDecoderLayer layer(2, 1, eps, kDevice);
    auto parameters = layer.parameters();
    struct ParameterCase {
        const char* name;
        std::vector<int> shape;
        std::vector<float> values;
    };
    const ParameterCase cases[] = {
        {"attention norm", {2}, {1, 0.75f}},
        {"Q", {2, 2}, {0.75f, -0.125f, 0.25f, 0.875f}},
        {"K", {2, 2}, {0.5f, 0.25f, -0.25f, 0.75f}},
        {"V", {2, 2}, {0.75f, -0.5f, 0.25f, 0.5f}},
        {"attention projection", {2, 2}, {0.75f, 0.25f, -0.25f, 0.5f}},
        {"MLP norm", {2}, {0.875f, 1.125f}},
        {"expand", {8, 2}, {0.5f, -0.25f, 0.25f, 0.625f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"contract", {2, 8}, {0.625f, 0.125f, 0, 0, 0, 0, 0, 0, -0.375f, 0.5f, 0, 0, 0, 0, 0, 0}},
    };
    Expect(parameters.size() == 8, "decoder must register eight parameters");
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const auto& test = cases[i];
        Expect(parameters[i].shape() == test.shape && parameters[i].numel() == test.values.size() &&
                   parameters[i].requires_grad(), "decoder parameter shape or registration");
        parameters[i].copy_(Tensor(test.values.data(), test.shape, kFloat, kDevice));
    }

    const float inputs[] = {0.5f, -0.25f, -0.375f, 0.75f};
    const float upstream[] = {-0.25f, 0.5f, 1.0f, -0.375f};
    Tensor input(inputs, {2, 2}, kFloat, kDevice);
    input.requires_grad_(true);
    Tensor output = layer(input);
    Expect(output.shape() == input.shape() && output.dtype() == kFloat && output.device() == kDevice,
           "decoder output metadata");
    // 独立 FP64 参考：首个 token 只看自己，第二个以 RoPE 角度 1 看两个 token。
    ExpectNear(output, {1.68420063f, -0.69196362f, -0.57009009f, 1.25355682f}, "decoder forward");
    output.backward(Tensor(upstream, {2, 2}, kFloat, kDevice));

    NogradNode guard;
    const auto objective = [&] {
        const auto values = Values(layer(input));
        double total = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            total += static_cast<double>(values[i]) * upstream[i];
        }
        return total;
    };
    const auto check_gradient = [&](Tensor& tensor, const char* name) {
        const Tensor grad = tensor.grad();
        const std::string message = std::string("decoder gradient: ") + name;
        Expect(!grad.empty() && grad.shape() == tensor.shape() && grad.dtype() == kFloat &&
                   grad.device() == kDevice, message.c_str());
        const auto values = Values(grad);
        std::size_t index = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            Expect(std::isfinite(values[i]), message.c_str());
            if (std::abs(values[i]) > std::abs(values[index])) index = i;
        }
        // 每份梯度选幅值最大的元素做中心差分，避免在接近零的位置验证连通性。
        Expect(std::abs(values[index]) > 1e-4f, message.c_str());
        Tensor element = tensor.reshape({-1})[static_cast<int>(index)];
        const float original = Values(element).front();
        constexpr float step = 1e-3f;
        const float plus = original + step;
        const float minus = original - step;
        element.fill_(plus);
        const double upper = objective();
        element.fill_(minus);
        const double lower = objective();
        element.fill_(original);
        const double estimate = (upper - lower) / (static_cast<double>(plus) - minus);
        Expect(std::abs(values[index] - estimate) <= 3e-4 + 1e-3 * std::abs(estimate), message.c_str());
    };
    check_gradient(input, "input");
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        check_gradient(parameters[i], cases[i].name);
    }
}

void CheckCrossEntropy() {
    const float logit_data[] = {1000, 1001, 1002, -1000, -1000, -1000};
    const std::int32_t label_data[] = {1, 0};
    Tensor logits(logit_data, {2, 3}, kFloat, kDevice);
    Tensor labels(label_data, {2}, Dtype::kInt32, kDevice);
    logits.requires_grad_(true);
    Tensor loss = cross_entropy(logits, labels);
    const float e1 = std::exp(-1.0f);
    const float e2 = std::exp(-2.0f);
    const float partition = 1 + e1 + e2;
    Expect(loss.is_scalar(), "mean cross entropy is scalar");
    ExpectNear(loss, {(1 + std::log(partition) + std::log(3.0f)) / 2}, "stable cross entropy");
    loss.backward(full({}, 2, kFloat, kDevice));
    ExpectNear(logits.grad(),
               {e2 / partition, e1 / partition - 1, 1 / partition, -2.0f / 3, 1.0f / 3, 1.0f / 3},
               "cross entropy gradient, mean and upstream scale");
    labels.fill_(0);
    ExpectError<std::logic_error>([&] { loss.backward(); }, "cross entropy target version");
}

void CheckCrossEntropyStrided() {
    const float values[] = {1000, -1000, 8, 1001, -1000, 8, 1002, 1000, 8, 997, 1000, 8};
    const std::int32_t ids[] = {2, -1, 0, -1, 1, -1};
    Tensor source(values, {4, 3}, kFloat, kDevice);
    source.requires_grad_(true);
    const Tensor logits = source.transpose(0, 1);
    const Tensor label_storage(ids, {3, 2}, Dtype::kInt32, kDevice);
    const Tensor labels = label_storage.narrow(1, 0, 1).squeeze(1);
    Tensor loss = cross_entropy(logits, labels);
    loss.backward(full({}, -5.0f, kFloat, kDevice));

    const float e1 = std::exp(-1.0f);
    const float e2 = std::exp(-2.0f);
    const float e5 = std::exp(-5.0f);
    const float partition = e2 + e1 + 1.0f + e5;
    const float expected_loss = (std::log(partition) + (2000.0f + std::log(2.0f)) + std::log(4.0f)) / 3;
    ExpectNear(loss, {expected_loss}, "cross entropy forward with an underflowed target probability", 2e-4f);
    const float scale = -5.0f / 3.0f;
    ExpectNear(source.grad().transpose(0, 1),
               {e2 / partition * scale, e1 / partition * scale,
                (1.0f / partition - 1.0f) * scale, e5 / partition * scale,
                -scale, 0.0f, 0.5f * scale, 0.5f * scale,
                0.25f * scale, -0.75f * scale, 0.25f * scale, 0.25f * scale},
               "cross entropy gradient with strided inputs and extreme logits", 3e-7f);
}

}  // namespace

int main() {
#ifdef TRANSFORMER_TEST_CUDA
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
        return 77;
    }
    if (status != cudaSuccess) {
        std::cerr << cudaGetErrorString(status) << '\n';
        return 1;
    }
#endif
    try {
        CheckEmbedding();
        CheckEmbeddingLayer();
        CheckDecoderDimensions();
        CheckDecoderForwardBackward();
        CheckRoPE();
        CheckCrossEntropy();
        CheckCrossEntropyStrided();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Transformer operator tests passed\n";
}
