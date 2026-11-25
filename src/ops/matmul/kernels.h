#pragma once

#include <cstddef>

#include "mini_infer/tensor/layout_view.h"
#include "mini_infer/tensor/dtype.h"

namespace mtensor {

// Two-dimensional inputs with element strides and contiguous output.
void MatmulCPU(const void* a, const void* b, void* output, Dtype dtype,
               int m, int k, int n, StrideView a_stride,
               StrideView b_stride);

void MatmulCUDA(const void* a, const void* b, void* output, Dtype dtype,
                int m, int k, int n, StrideView a_stride,
                StrideView b_stride);

}  // namespace mtensor
