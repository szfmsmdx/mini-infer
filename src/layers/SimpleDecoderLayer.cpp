#include "mini_infer/layers/SimpleDecoderLayer.h"
#include "mini_infer/ops/activation.h"
#include "mini_infer/ops/attention.h"
#include "mini_infer/ops/position_embedding.h"
#include "ops/common/attention_shape.h"

namespace mtensor {

SimpleDecoderLayer::SimpleDecoderLayer(int channels, int heads, float norm_eps, Device device,
                                       const WeightInitializer& initializer,
                                       const WeightInitializer& residual_initializer)
    : heads_(CheckedDecoderHeads(channels, heads,
                                "SimpleDecoderLayer expects positive dimensions and an even head width")),
      attention_norm_(channels, norm_eps, device),
      q_(channels, channels, false, device, initializer),
      k_(channels, channels, false, device, initializer),
      v_(channels, channels, false, device, initializer),
      attention_projection_(channels, channels, false, device, residual_initializer),
      mlp_norm_(channels, norm_eps, device),
      expand_(channels, 4 * channels, false, device, initializer),
      contract_(4 * channels, channels, false, device, residual_initializer) {}

void SimpleDecoderLayer::append_child_parameters(std::vector<Tensor>& result) const {
    append_parameters(attention_norm_, result);
    append_parameters(q_, result);
    append_parameters(k_, result);
    append_parameters(v_, result);
    append_parameters(attention_projection_, result);
    append_parameters(mlp_norm_, result);
    append_parameters(expand_, result);
    append_parameters(contract_, result);
}

Tensor SimpleDecoderLayer::forward(const Tensor& x) {
    const Tensor attention_input = attention_norm_(x);
    const Tensor q = rope(q_(attention_input), heads_);
    const Tensor k = rope(k_(attention_input), heads_);
    const Tensor v = v_(attention_input);
    const Tensor attention = causal_attention(q, k, v, heads_);
    const Tensor residual = x + attention_projection_(attention);

    const Tensor mlp_input = mlp_norm_(residual);
    const Tensor expanded = expand_(mlp_input);
    return residual + contract_(silu(expanded));
}

}  // namespace mtensor
