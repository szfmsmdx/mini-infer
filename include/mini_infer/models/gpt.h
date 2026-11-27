#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "mini_infer/modules/embedding.h"
#include "mini_infer/modules/module.h"
#include "mini_infer/modules/rmsnorm.h"
#include "mini_infer/models/layers/simple_decoder_layer.h"
#include "mini_infer/models/model_config.h"

namespace mtensor {

struct GPTConfig : ModelConfig {
    constexpr GPTConfig(int vocab_size = 0, int context_length = 32, int channels = 48,
                        int heads = 4, int layers = 2)
        : ModelConfig{vocab_size, context_length, channels, heads, layers} {}

    GPTConfig validated() const;
};

class GPT : public Module {
public:
    GPT(GPTConfig config, Device device, std::uint32_t seed = 1337);

    // 单条序列的 Int32 token [T]，返回每个位置的 logits [T, vocab_size]。
    Tensor forward(const Tensor& tokens) override;
    const GPTConfig& config() const { return config_; }

private:
    GPT(GPTConfig config, Device device, std::mt19937 rng);

    GPTConfig config_;
    std::shared_ptr<Embedding> token_embedding_;
    std::vector<std::shared_ptr<SimpleDecoderLayer>> decoder_layers_;
    std::shared_ptr<RMSNorm> final_norm_;
};

}  // namespace mtensor
