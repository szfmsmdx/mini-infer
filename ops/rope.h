#pragma once

#include "tensor/tensor.h"

namespace mtensor {

enum class RopeStyle {
    kHalf,         // GPT-NeoX 风格：前后两半配对
    kInterleaved,  // 相邻维度配对
};

// input: [tokens, heads, head_dim]
// positions: [tokens]，dtype 为 Int32 或 Int64。
// 对每个 head 的前 rotary_dim 个维度做 RoPE；rotary_dim 必须为偶数且
// 不超过 head_dim，剩余维度保持不变。
//
// 使用示例（x 为 shape {3, 16, 128}，每个 token 一个位置）：
//     int32_t pos_data[] = {0, 1, 2};
//     Tensor pos(pos_data, {3}, Dtype::kInt32);
//     Tensor y = rope(x, pos, 10000.0f, 64);   // 只旋转前 64 维
//     Tensor z = rope(x, pos, 10000.0f, 64, RopeStyle::kInterleaved);
Tensor rope(
    const Tensor& input,
    const Tensor& positions,
    float theta,
    int rotary_dim,
    RopeStyle style = RopeStyle::kHalf);

// TODO：Qwen3.5 的 MRoPE 还需要位置网格和 sections，等视觉路径明确后补充接口。

}  // namespace mtensor