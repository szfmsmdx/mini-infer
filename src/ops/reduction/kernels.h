#pragma once

#include <cstddef>
#include <vector>

#include "mini_infer/dtype.h"

namespace mtensor {

void SumCPU(const void* input, void* output, Dtype dtype, const std::vector<int>& shape,
            const std::vector<std::size_t>& stride, std::size_t numel);

void SumCPU(const void* input, void* output, Dtype dtype, const std::vector<int>& shape,
            const std::vector<std::size_t>& stride, std::size_t numel, int dim);

// input、output、shape 和 stride 均为设备指针；output 指向一个标量。
void SumCUDA(const void* input, void* output, Dtype dtype, const int* shape,
             const std::size_t* stride, int ndim, std::size_t numel);

// 按维度求和，output 指向连续输出，output_numel 为输出元素个数。
void SumCUDA(const void* input, void* output, Dtype dtype, const int* shape,
             const std::size_t* stride, int ndim, std::size_t output_numel, int dim);

}  // namespace mtensor
