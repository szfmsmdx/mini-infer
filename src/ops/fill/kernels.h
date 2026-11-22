#pragma once

#include <cstddef>
#include <vector>

#include "mini_infer/dtype.h"

namespace mtensor {

void ZeroCPU(void* data_ptr, const std::vector<int>& shape,
             const std::vector<std::size_t>& stride, std::size_t element_size);

void FillCPU(void* data_ptr, const std::vector<int>& shape,
             const std::vector<std::size_t>& stride, std::size_t element_size,
             const void* host_value);

void ZeroCUDA(void* data_ptr, const int* shape_cuda, const std::size_t* stride_cuda,
              int ndim, std::size_t numel, std::size_t element_size);

void FillCUDA(void* device_ptr, std::size_t numel, Dtype dtype, const void* host_value,
              const int* shape, const std::size_t* stride, int ndim);

void FillCUDAFromDevice(void* device_ptr, std::size_t numel, std::size_t element_size,
                        const void* device_value, const int* shape,
                        const std::size_t* stride, int ndim);

}  // namespace mtensor
