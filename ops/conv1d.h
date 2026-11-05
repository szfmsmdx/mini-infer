#pragma once

#include "tensor/tensor.h"

namespace mtensor {

// 深度可分离因果一维卷积，供 Gated DeltaNet 使用。
// input: [tokens, channels]
// weight: [channels, kernel_size]，每个通道独立卷积。
// bias: 可为 nullptr；非空时 shape 为 [channels]。
// 输出 shape 与 input 相同；左侧按 kernel_size - 1 做因果补零。
//
// 使用示例（x 为 shape {T, C}，卷积核宽度为 4）：
//     Tensor w(w_data, {C, 4}, Dtype::kFloat32);
//     Tensor y = depthwise_causal_conv1d(x, w);
//     Tensor y2 = depthwise_causal_conv1d(x, w, &bias);
Tensor depthwise_causal_conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr);

}  // namespace mtensor