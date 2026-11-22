#pragma once

#include <cstddef>
#include <vector>

namespace mtensor {

class Tensor;

// 借用数据指针，存储所有权仍由 Tensor 管理。
void CopyCPU(void* dst_data_ptr, const void* src_data_ptr, const std::vector<int>& shape,
             const std::vector<std::size_t>& dst_stride,
             const std::vector<std::size_t>& src_stride, std::size_t element_size);

void CopyCUDA(void* destination, const void* source, const int* shape,
              const std::size_t* dst_stride, const std::size_t* src_stride,
              int ndim, std::size_t numel, std::size_t element_size);

void CopyHostToCUDA(void* destination, const void* source, std::size_t bytes);

void CopyCUDAToHost(void* destination, const void* source, std::size_t bytes);

}  // namespace mtensor
