#pragma once

#include <utility>

#include "tensor/tensor.h"

namespace mtensor {

// dim 支持负数；keepdim 为 true 时保留被归约的维度并设为 1。
//
// 使用示例（x 为 shape {2, 3}）：
//     Tensor s = sum(x, 1);          // {2}
//     Tensor m = mean(x, 1, true);   // {2, 1}
//     Tensor v = max(x, 1);          // {2}
Tensor sum(const Tensor& input, int dim, bool keepdim = false);
Tensor mean(const Tensor& input, int dim, bool keepdim = false);
Tensor max(const Tensor& input, int dim, bool keepdim = false);

// 在 dim 上做数值稳定的 softmax。
//
// 使用示例：
//     Tensor p = softmax(x, 1);      // 每一行和为 1
Tensor softmax(const Tensor& input, int dim = -1);

// 返回 Int64 索引；keepdim 为 true 时保留被归约的维度并设为 1。
//
// 使用示例：
//     Tensor idx = argmax(x, 1);     // {2}，每个最大值的位置
Tensor argmax(const Tensor& input, int dim = -1, bool keepdim = false);

// 沿 dim 取前 k 个元素，values 按降序排列；indices 为 Int64。
//
// 使用示例：
//     auto [values, indices] = topk(x, 2, 1);  // values/indices: {2, 2}
std::pair<Tensor, Tensor> topk(const Tensor& input, int k, int dim = -1);

}  // namespace mtensor