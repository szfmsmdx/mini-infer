#pragma once

#include <cstddef>

#include "tensor/dtype.h"
#include "tensor/tensor.h"

namespace mtensor {

// 输入的 shape、dtype、device 必须相同；返回独立存储，不修改输入。
// 不做广播或隐式类型转换；尚未实现的 dtype/device 必须明确报错。
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& a);

}  // namespace mtensor
