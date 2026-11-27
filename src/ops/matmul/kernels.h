#pragma once

#include <cstddef>

#include "mini_infer/tensor/layout_view.h"
#include "mini_infer/tensor/dtype.h"

namespace mtensor {

// 输入是二维矩阵，步长以元素为单位；输出连续存储。
void MatmulCPU(const void* a, const void* b, void* output, Dtype dtype,
               int m, int k, int n, StrideView a_stride,
               StrideView b_stride);

void MatmulCUDA(const void* a, const void* b, void* output, Dtype dtype,
                int m, int k, int n, StrideView a_stride,
                StrideView b_stride);

}  // namespace mtensor
