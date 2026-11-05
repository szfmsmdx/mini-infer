#pragma once

#include "tensor/tensor.h"

namespace mtensor {

// indices 的 dtype 必须是 Int32 或 Int64。
// weight[vocab, hidden] -> indices.shape + [hidden]。
//
// 使用示例：
//     int64_t id_data[] = {3, 7};
//     Tensor ids(id_data, {2}, Dtype::kInt64);
//     Tensor e = embedding(weight, ids);      // weight: {V,H} -> {2,H}
Tensor embedding(const Tensor& weight, const Tensor& indices);

// 沿 dim 按 indices 选择切片；输出 dtype/device 与 input 相同。
//
// 使用示例：
//     Tensor part = index_select(x, 0, ids);  // 按 ids 选第 0 维的行
Tensor index_select(const Tensor& input, int dim, const Tensor& indices);

// 按 indices 逐元素取值，输出 shape 与 indices 相同。
//
// 使用示例：
//     Tensor g = gather(x, 1, index);         // 沿第 1 维逐元素取值
Tensor gather(const Tensor& input, int dim, const Tensor& indices);

// 返回新的 Tensor，不修改 input：
// output = input，然后把 src 按 indices 累加到 output。
//
// 使用示例：
//     Tensor s = scatter_add(x, 0, ids, src); // 把 src 的行累加到 x 的副本
Tensor scatter_add(
    const Tensor& input,
    int dim,
    const Tensor& indices,
    const Tensor& src);

}  // namespace mtensor