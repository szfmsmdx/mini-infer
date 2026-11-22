#pragma once

#include "mini_infer/tensor.h"

namespace mtensor {

// Q、K、V 均为 FP32 [序列, 通道]，缩放点积注意力仅看当前位置及之前的位置。
Tensor causal_attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads);

}  // namespace mtensor
