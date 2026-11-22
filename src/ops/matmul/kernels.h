#pragma once

#include <cstddef>
#include <vector>

#include "mini_infer/dtype.h"

namespace mtensor {

// Two-dimensional inputs with element strides and contiguous output.
void MatmulCPU(const void* a, const void* b, void* output, Dtype dtype,
               int m, int k, int n, const std::vector<std::size_t>& a_stride,
               const std::vector<std::size_t>& b_stride);

void MatmulCUDA(const void* a, const void* b, void* output, Dtype dtype,
                int m, int k, int n, const std::vector<std::size_t>& a_stride,
                const std::vector<std::size_t>& b_stride);

}  // namespace mtensor
