#pragma once

#include <cstddef>

#include "backend/common/cuda_layout.h"
#include "mini_infer/tensor/layout_view.h"

namespace mtensor {

class Tensor;

// 借用数据指针，存储所有权仍由 Tensor 管理。
void CopyCPU(void* dst_data_ptr, const void* src_data_ptr, ShapeView shape,
             StrideView dst_stride,
             StrideView src_stride, std::size_t element_size);

void CopyCUDA(void* destination, const void* source, CUDALayout dst_layout, CUDALayout src_layout, std::size_t numel, std::size_t element_size);

void CastCPU(const Tensor& input, Tensor& output);
void CastCUDA(const Tensor& input, Tensor& output);

}  // namespace mtensor
