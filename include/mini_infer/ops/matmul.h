#pragma once

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

Tensor matmul(const Tensor& a, const Tensor& b);
// 输出须是独立的连续 Tensor，shape 为 [a.size(0), b.size(1)]。
// 写入前递增输出版本；不创建反向节点，也不替换或清除已有节点。
// 调用方须保证这次覆盖不会影响后续反向计算。
Tensor& matmul_out(const Tensor& a, const Tensor& b, Tensor& out);

}  // namespace mtensor
