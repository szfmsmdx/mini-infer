#pragma once

#include "mini_infer/core/tensor.h"

namespace mtensor {

// 对全部元素求和，返回标量 Tensor。
Tensor sum(const Tensor& input);

// 沿指定维度求和；dim 支持负索引，keepdim 为 true 时保留该维度且大小为 1。
Tensor sum(const Tensor& input, int dim, bool keepdim = false);

// 浮点均值，输出保留输入 dtype 和 device。
Tensor mean(const Tensor& input);
Tensor mean(const Tensor& input, int dim, bool keepdim = false);

// 只返回极值，不返回索引。保留 dtype 和 device；并列极值平分梯度。
Tensor max(const Tensor& input);
Tensor max(const Tensor& input, int dim, bool keepdim = false);
Tensor min(const Tensor& input);
Tensor min(const Tensor& input, int dim, bool keepdim = false);

}  // namespace mtensor
