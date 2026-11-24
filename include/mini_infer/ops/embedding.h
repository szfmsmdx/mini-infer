#pragma once

#include "mini_infer/core/tensor.h"

namespace mtensor {

// 从 FP32 权重 [词表, 通道] 查表，Int32 token 输入 [序列]，输出 [序列, 通道]。
Tensor embedding(const Tensor& weight, const Tensor& indices);

}  // namespace mtensor
