#pragma once

#include <cstddef>
#include <vector>

#include "mini_infer/dtype.h"

namespace mtensor {

class Tensor;

void ExpCPU(const Tensor& x, Tensor& out);

void ExpCUDA(const Tensor& x, Tensor& out);

void LogCPU(const Tensor& x, Tensor& out);

void LogCUDA(const Tensor& x, Tensor& out);

void RsqrtCPU(const Tensor& x, Tensor& out);

void RsqrtCUDA(const Tensor& x, Tensor& out);

// 输入按 stride 读取，输出连续；整数输入对应 Float32 输出。
void SqrtCPU(const void* input, void* output, Dtype dtype, const std::vector<int>& shape,
             const std::vector<std::size_t>& stride);

void SqrtCUDA(const void* input, void* output, Dtype dtype, std::size_t numel,
              const int* shape, const std::size_t* stride, int ndim);

// CPU 后端入口，在 ops/elementwise/cpu.cpp 中按 dtype 调用内部模板。
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

// 在 ops/elementwise/cuda.cu 中实现的普通 CPU 函数，负责启动 CUDA kernel。
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

// 内部 CUDA 后端：按各自 stride 读取 x 和上游梯度，写入连续 dx。
void RsqrtBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);

}  // namespace mtensor
