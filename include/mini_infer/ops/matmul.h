#pragma once

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

Tensor matmul(const Tensor& a, const Tensor& b);
// 输出须是独立的连续 Tensor，shape 为 [a.size(0), b.size(1)]。
Tensor& matmul_out(const Tensor& a, const Tensor& b, Tensor& out);

}  // namespace mtensor
