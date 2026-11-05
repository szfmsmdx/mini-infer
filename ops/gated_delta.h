#pragma once

#include "tensor/tensor.h"

namespace mtensor {

// Gated DeltaNet 的循环更新，不包含外层投影、卷积和 gated RMSNorm。
// q/k: [tokens, value_heads, key_dim]，函数内做 L2 归一化和 1/sqrt(key_dim) 缩放。
// v:   [tokens, value_heads, value_dim]
// beta/g: [tokens, value_heads]
// state: [value_heads, key_dim, value_dim]，读写；prefill 前初始化为 0，
//        decode 时持续复用。
// 返回 [tokens, value_heads, value_dim]。
//
// 使用示例（prefill：state 初始化为 0；decode：复用同一个 state）：
//     Tensor state({H, Dk, Dv}, Dtype::kFloat32);
//     Tensor out = gated_delta_rule(q, k, v, beta, g, state);
Tensor gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& g,
    Tensor& state);

}  // namespace mtensor