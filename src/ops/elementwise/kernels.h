#pragma once

#include <cstddef>

#include "ops/common/cuda_layout.h"
#include "mini_infer/core/layout_view.h"
#include "mini_infer/core/dtype.h"

namespace mtensor {

class Tensor;

void ExpCPU(const Tensor& x, Tensor& out);

void ExpCUDA(const Tensor& x, Tensor& out);

void LogCPU(const Tensor& x, Tensor& out);

void LogCUDA(const Tensor& x, Tensor& out);

void RsqrtCPU(const Tensor& x, Tensor& out);

void RsqrtCUDA(const Tensor& x, Tensor& out);

// 输入按 stride 读取，输出连续；整数输入对应 Float32 输出。
void SqrtCPU(const void* input, void* output, Dtype dtype, ShapeView shape,
             StrideView stride);

void SqrtCUDA(const void* input, void* output, Dtype dtype, std::size_t numel,
              CUDALayout layout);

// CPU 后端入口，在 ops/elementwise/cpu.cpp 中按 dtype 调用内部模板。
// 输入和输出均为主机指针，stride 的单位是元素；不接管指针的所有权。
void AddCPU(const void* a, const void* b, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView b_stride, StrideView output_stride);

void SubCPU(const void* a, const void* b, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView b_stride, StrideView output_stride);

void MulCPU(const void* a, const void* b, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView b_stride, StrideView output_stride);

void DivCPU(const void* a, const void* b, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView b_stride, StrideView output_stride);

void NegCPU(const void* a, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView output_stride);

// 在 ops/elementwise/cuda.cu 中实现的普通 CPU 函数，负责启动 CUDA kernel。
// 输入和输出均为设备指针，numel 为元素个数；不接管指针的所有权。
void AddCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout, CUDALayout b_layout);

void SubCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout, CUDALayout b_layout);

void MulCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout, CUDALayout b_layout);

void DivCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout, CUDALayout b_layout);

void NegCUDA(const void* a, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout);

// 内部 CUDA 后端：按各自 stride 读取 x 和上游梯度，写入连续 dx。
void RsqrtBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);

}  // namespace mtensor
