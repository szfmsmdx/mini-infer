#pragma once

#include "tensor/tensor.h"

namespace mtensor {

// 因果自注意力核心，不含 QK RMSNorm、RoPE 和输出门。
// query: [query_tokens, query_heads, head_dim]
// key/value: [key_tokens, kv_heads, head_dim]
// query_heads 必须是 kv_heads 的整数倍，按 GQA 分组共享 KV。
// 返回 [query_tokens, query_heads, head_dim]。
//
// 使用示例（16 个 Q head 共享 4 个 KV head）：
//     Tensor out = causal_attention(q, k, v, 1.0f / 16.0f);
Tensor causal_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    float scale);

// TODO：padding mask、KV cache 读取接口和 batch 维度等到模型运行时确定。

}  // namespace mtensor