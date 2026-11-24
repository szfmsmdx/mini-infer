#pragma once

#include "mini_infer/core/tensor.h"

namespace mtensor {

// 沿 x 的最后一维归一化；weight 是与该维等长的一维权重。
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);

}  // namespace mtensor
