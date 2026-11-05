#pragma once

#include "tensor/tensor.h"

namespace mtensor {

// input[..., hidden] 在最后一维做 RMSNorm；weight 的 shape 为 [hidden]。
//
// 使用示例（x 为 shape {2, 4}）：
//     Tensor w(w_data, {4}, Dtype::kFloat32);
//     Tensor y = rms_norm(x, w, 1e-6f);         // {2, 4}
Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon);

// Qwen 使用的零中心 RMSNorm：缩放系数为 (1 + weight)。
//
// 使用示例：
//     Tensor y = rms_norm_zero_centered(x, w, 1e-6f);
Tensor rms_norm_zero_centered(
    const Tensor& input,
    const Tensor& weight,
    float epsilon);

// output = rms_norm(input, weight, epsilon) * silu(gate)。
// gate 与 input 的 shape、dtype、device 相同。
//
// 使用示例：
//     Tensor gate(gate_data, {2, 4}, Dtype::kFloat32);
//     Tensor y = rms_norm_gated(x, w, gate, 1e-6f);
Tensor rms_norm_gated(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& gate,
    float epsilon);

// 标准 LayerNorm，主要供视觉编码器使用；weight、bias 的 shape 为 [hidden]。
//
// 使用示例：
//     Tensor bias(bias_data, {4}, Dtype::kFloat32);
//     Tensor y = layer_norm(x, w, bias, 1e-5f);
Tensor layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    float epsilon);

}  // namespace mtensor