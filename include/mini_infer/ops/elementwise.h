#pragma once

#include <cstddef>

#include "mini_infer/core/dtype.h"
#include "mini_infer/core/tensor.h"

namespace mtensor {

// 输入的 shape、dtype、device 必须相同；返回独立存储，不修改输入。
// 不做广播或隐式类型转换；尚未实现的 dtype/device 必须明确报错。
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& a);

// 浮点输入保持 dtype；整数先转 Float32。负数的实平方根为 NaN。
Tensor sqrt(const Tensor& x);
Tensor rsqrt(const Tensor& x);

// 逐元素计算自然指数；整数输入返回 Float32。
Tensor exp(const Tensor& x);

// 逐元素计算自然对数；整数输入返回 Float32。零返回 -Inf，负数返回 NaN。
Tensor log(const Tensor& x);

}  // namespace mtensor
