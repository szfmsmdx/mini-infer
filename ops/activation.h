#pragma once

#include "tensor/tensor.h"

namespace mtensor {

// 逐元素激活函数，输出与输入的 shape、dtype、device 相同。
//
// 使用示例（x 为任意 shape 的 Tensor）：
//     Tensor y = silu(x);
//     Tensor s = sigmoid(x);
//     Tensor p = softplus(x);
//     Tensor e = exp(x);
Tensor silu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor softplus(const Tensor& input);
Tensor exp(const Tensor& input);

}  // namespace mtensor