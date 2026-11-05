#pragma once

#include "tensor/tensor.h"

namespace mtensor {

// 二维矩阵乘法：a[M, K] × b[K, N] -> [M, N]。
// 输入必须非空，shape、dtype、device 匹配；暂不支持 batch 和广播。
//
// 使用示例：
//     Tensor a(a_data, {2, 3}, Dtype::kFloat32);
//     Tensor b(b_data, {3, 4}, Dtype::kFloat32);
//     Tensor c = matmul(a, b);                 // {2, 4}
Tensor matmul(const Tensor& a, const Tensor& b);

// 线性层：input[rows, in_features] × weight[out_features, in_features]^T
// -> [rows, out_features]。bias 可为 nullptr；非空时 shape 为 [out_features]。
//
// 使用示例：
//     Tensor w(w_data, {8, 3}, Dtype::kFloat32);
//     Tensor y = linear(a, w);                  // a: {2,3} -> {2,8}
//     Tensor bias(bias_data, {8}, Dtype::kFloat32);
//     Tensor z = linear(a, w, &bias);           // {2, 8}
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor* bias = nullptr);

}  // namespace mtensor