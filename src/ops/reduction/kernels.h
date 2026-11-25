#pragma once

#include <cstddef>

#include "ops/common/cuda_layout.h"
#include "mini_infer/tensor/layout_view.h"
#include "mini_infer/tensor/dtype.h"

namespace mtensor {

class Tensor;

void SumCPU(const void* input, void* output, Dtype dtype, ShapeView shape,
            StrideView stride, std::size_t numel);

void SumCPU(const void* input, void* output, Dtype dtype, ShapeView shape,
            StrideView stride, std::size_t numel, int dim);

// input 和 output 为设备指针，布局按值传入；output 指向一个标量。
void SumCUDA(const void* input, void* output, Dtype dtype, CUDALayout layout, std::size_t numel);

// 按维度求和，output 指向连续输出，output_numel 为输出元素个数。
void SumCUDA(const void* input, void* output, Dtype dtype, CUDALayout layout, std::size_t output_numel, int dim);

// inner_size 是归约维之后的逻辑元素数；全量归约时为 1。
void ExtremaCPU(const Tensor& input, Tensor& output, std::size_t inner_size, bool maximum);
void ExtremaCUDA(const Tensor& input, Tensor& output, std::size_t inner_size, bool maximum);
void ExtremaGradCPU(const Tensor& input, const Tensor& extrema, const Tensor& grad,
                    Tensor& input_grad, std::size_t inner_size);
void ExtremaGradCUDA(const Tensor& input, const Tensor& extrema, const Tensor& grad,
                     Tensor& input_grad, std::size_t inner_size);

}  // namespace mtensor
