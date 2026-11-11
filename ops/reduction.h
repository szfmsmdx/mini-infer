#pragma once

#include "tensor/tensor.h"

namespace mtensor {

// 对全部元素求和，返回标量 Tensor。
Tensor sum(const Tensor& input);

}  // namespace mtensor
