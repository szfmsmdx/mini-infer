#pragma once

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

Tensor sigmoid(const Tensor& x);

// 逐元素计算 x * sigmoid(x)，输出形状与 x 相同；整数输入返回 FP32。
Tensor silu(const Tensor& x);

// 沿 dim 归一化概率，内部使用 FP32。浮点输出保留 dtype，整数输出 FP32。
Tensor softmax(const Tensor& x, int dim = -1);

}  // namespace mtensor
