#pragma once

#include <cstddef>

#include "ops/common/cuda_layout.h"
#include "mini_infer/tensor/layout_view.h"
#include "mini_infer/tensor/dtype.h"

namespace mtensor {

void ZeroCPU(void* data_ptr, ShapeView shape,
             StrideView stride, std::size_t element_size);

void FillCPU(void* data_ptr, ShapeView shape,
             StrideView stride, std::size_t element_size,
             const void* host_value);

void ZeroCUDA(void* data_ptr, CUDALayout layout, std::size_t numel, std::size_t element_size);

void FillCUDA(void* device_ptr, std::size_t numel, Dtype dtype, const void* host_value,
              CUDALayout layout);

void FillCUDAFromDevice(void* device_ptr, std::size_t numel, std::size_t element_size,
                        const void* device_value, CUDALayout layout);

}  // namespace mtensor
