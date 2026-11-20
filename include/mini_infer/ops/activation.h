#pragma once

#include "mini_infer/tensor.h"

namespace mtensor {

Tensor sigmoid(const Tensor& x);

// 逐元素计算 x * sigmoid(x)，输出形状与 x 相同；整数输入返回 FP32。
Tensor silu(const Tensor& x);

}  // namespace mtensor
