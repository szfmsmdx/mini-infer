#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include "mini_infer/models/gpt.h"
#include "mini_infer/init.h"
#include "ops/common/attention_shape.h"

namespace mtensor {
namespace {
constexpr float kNormEps = 1e-5f;
constexpr float kWeightStddev = 0.02f;

}  // namespace

GPTConfig GPTConfig::validated() const {
    constexpr const char* message =
        "GPT expects a vocabulary of at least two, positive dimensions, and an even head width";
    check_positive_dimensions(message);
    if (vocab_size < 2) throw std::invalid_argument(message);
    CheckedDecoderHeads(channels, heads, message);
    return *this;
}

GPT::GPT(GPTConfig config, Device device, std::uint32_t seed)
    : GPT(config, device, std::mt19937(seed)) {}

GPT::GPT(GPTConfig config, Device device, std::mt19937 rng)
    : config_(config.validated()) {
    token_embedding_ = register_module("token_embedding", std::make_shared<Embedding>(
        config_.vocab_size, config_.channels, device,
        weight_init::make_normal_initializer(rng, kWeightStddev)));
    final_norm_ = std::make_shared<RMSNorm>(config_.channels, kNormEps, device);
    const auto initializer = weight_init::make_normal_initializer(rng, kWeightStddev);
    const auto residual_initializer =
        weight_init::make_normal_initializer(rng, kWeightStddev / std::sqrt(2.0f * config_.layers));
    decoder_layers_.reserve(static_cast<std::size_t>(config_.layers));
    for (int layer = 0; layer < config_.layers; ++layer) {
        auto decoder = std::make_shared<SimpleDecoderLayer>(
            config_.channels, config_.heads, kNormEps, device, initializer, residual_initializer);
        decoder_layers_.push_back(register_module("decoder_" + std::to_string(layer), decoder));
    }
    // checkpoint 按顺序存权重，最终归一化必须在 Decoder 之后登记。
    register_module("final_norm", final_norm_);
}

Tensor GPT::forward(const Tensor& tokens) {
    if (tokens.dim() != 1 || tokens.size(0) > config_.context_length) {
        throw std::invalid_argument("GPT expects one token sequence within context_length");
    }

    Tensor x = token_embedding_->forward(tokens);
    for (const auto& layer : decoder_layers_) {
        x = layer->forward(x);
    }
    // 输出投影与词嵌入共享参数，两路梯度累加到同一份权重。
    return final_norm_->forward(x).matmul(token_embedding_->weight().transpose(0, 1));
}

}  // namespace mtensor
