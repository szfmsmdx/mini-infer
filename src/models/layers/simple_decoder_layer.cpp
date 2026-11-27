#include <memory>

#include "mini_infer/models/layers/simple_decoder_layer.h"
#include "mini_infer/ops/activation.h"
#include "mini_infer/ops/attention.h"
#include "mini_infer/ops/position_embedding.h"
#include "ops/common/attention_shape.h"

namespace mtensor {

SimpleDecoderLayer::SimpleDecoderLayer(int channels, int heads, float norm_eps, Device device,
                                       const WeightInitializer& initializer,
                                       const WeightInitializer& residual_initializer)
    : heads_(CheckedDecoderHeads(channels, heads,
                                "SimpleDecoderLayer expects positive dimensions and an even head width")) {
    attention_norm_ = register_module("attention_norm", std::make_shared<RMSNorm>(channels, norm_eps, device));
    q_ = register_module("q", std::make_shared<Linear>(channels, channels, false, device, initializer));
    k_ = register_module("k", std::make_shared<Linear>(channels, channels, false, device, initializer));
    v_ = register_module("v", std::make_shared<Linear>(channels, channels, false, device, initializer));
    attention_projection_ = register_module(
        "attention_projection",
        std::make_shared<Linear>(channels, channels, false, device, residual_initializer));
    mlp_norm_ = register_module("mlp_norm", std::make_shared<RMSNorm>(channels, norm_eps, device));
    expand_ = register_module("expand", std::make_shared<Linear>(channels, 4 * channels, false, device, initializer));
    contract_ = register_module(
        "contract", std::make_shared<Linear>(4 * channels, channels, false, device, residual_initializer));
}

Tensor SimpleDecoderLayer::forward(const Tensor& x) {
    const Tensor attention_input = attention_norm_->forward(x);
    const Tensor q = rope(q_->forward(attention_input), heads_);
    const Tensor k = rope(k_->forward(attention_input), heads_);
    const Tensor v = v_->forward(attention_input);
    const Tensor attention = causal_attention(q, k, v, heads_);
    const Tensor residual = x + attention_projection_->forward(attention);

    const Tensor mlp_input = mlp_norm_->forward(residual);
    const Tensor expanded = expand_->forward(mlp_input);
    return residual + contract_->forward(silu(expanded));
}

}  // namespace mtensor
