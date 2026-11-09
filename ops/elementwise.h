#pragma once

#include <cstddef>

#include "tensor/dtype.h"
#include "tensor/tensor.h"

namespace mtensor {

// 输入的 shape、dtype、device 必须相同；返回独立存储，不修改输入。
// 不做广播或隐式类型转换；尚未实现的 dtype/device 必须明确报错。
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& a);

// CPU 后端入口，在 elementwise_cpu.cpp 中按 dtype 调用内部模板。
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

// 在 elementwise_cuda.cu 中实现的普通 CPU 函数，负责启动 CUDA kernel。
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

}  // namespace mtensor
