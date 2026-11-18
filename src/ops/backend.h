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

// CPU 后端入口，在 ops/cpu/elementwise.cpp 中按 dtype 调用内部模板。
// 输入和输出均为主机指针，stride 的单位是元素；不接管指针的所有权。
void AddCPU(const void* a, const void* b, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride);
void SubCPU(const void* a, const void* b, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride);
void MulCPU(const void* a, const void* b, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride);
void DivCPU(const void* a, const void* b, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride);
void NegCPU(const void* a, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& output_stride);

// 在 ops/cuda/elementwise.cu 中实现的普通 CPU 函数，负责启动 CUDA kernel。
// 输入和输出均为设备指针，numel 为元素个数；不接管指针的所有权。
void AddCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim);
void SubCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim);
void MulCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim);
void DivCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim);
void NegCUDA(const void* a, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, int ndim);

// 借用数据指针，存储所有权仍由 Tensor 管理。
void CopyCPU(void* dst_data_ptr, const void* src_data_ptr, const std::vector<int>& shape,
             const std::vector<std::size_t>& dst_stride,
             const std::vector<std::size_t>& src_stride, std::size_t element_size);
void CopyCUDA(void* destination, const void* source, const int* shape,
              const std::size_t* dst_stride, const std::size_t* src_stride,
              int ndim, std::size_t numel, std::size_t element_size);
void CopyHostToCUDA(void* destination, const void* source, std::size_t bytes);
void CopyCUDAToHost(void* destination, const void* source, std::size_t bytes);

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

void PrintCPU(const void* data_ptr, const std::vector<int>& shape,
              const std::vector<std::size_t>& stride, Dtype dtype, bool is_empty);

}  // namespace mtensor
