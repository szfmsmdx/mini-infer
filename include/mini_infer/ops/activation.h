#pragma once

#include "mini_infer/tensor.h"

namespace mtensor {

// 逐元素计算 x * sigmoid(x)，输出形状与 x 相同。
Tensor silu(const Tensor& x);

}  // namespace mtensor
