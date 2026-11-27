#pragma once

#include <cstddef>

#include "mini_infer/tensor/dtype.h"
#include "mini_infer/tensor/tensor.h"

namespace mtensor {

// 二元算子按广播规则匹配 shape；输入 dtype 和 device 必须相同。
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& a);

// 本文件的 *_out 写入调用方预分配的连续输出。输入需已对齐 shape，输出不能与输入共享 Storage。
// 写入前递增输出版本；不创建反向节点，也不替换或清除已有节点。
// 调用方须保证这次覆盖不会影响后续反向计算。
Tensor& add_out(const Tensor& a, const Tensor& b, Tensor& out);
Tensor& sub_out(const Tensor& a, const Tensor& b, Tensor& out);
Tensor& mul_out(const Tensor& a, const Tensor& b, Tensor& out);
Tensor& div_out(const Tensor& a, const Tensor& b, Tensor& out);
Tensor& neg_out(const Tensor& x, Tensor& out);

// 浮点输入保持 dtype；整数先转 Float32。负数的实平方根为 NaN。
Tensor sqrt(const Tensor& x);
Tensor rsqrt(const Tensor& x);
Tensor& sqrt_out(const Tensor& x, Tensor& out);
Tensor& rsqrt_out(const Tensor& x, Tensor& out);

// 逐元素计算自然指数；整数输入返回 Float32。
Tensor exp(const Tensor& x);
Tensor& exp_out(const Tensor& x, Tensor& out);

// 逐元素计算自然对数；整数输入返回 Float32。零返回 -Inf，负数返回 NaN。
Tensor log(const Tensor& x);
Tensor& log_out(const Tensor& x, Tensor& out);

}  // namespace mtensor
