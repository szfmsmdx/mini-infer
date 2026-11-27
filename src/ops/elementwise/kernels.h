#pragma once

#include <cstddef>

#include "backend/common/cuda_layout.h"
#include "mini_infer/tensor/layout_view.h"
#include "mini_infer/tensor/dtype.h"

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

// CPU 后端按 dtype 调用内部模板。
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

// CUDA 后端的主机函数负责启动 kernel。
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

void SubBackwardCPU(const Tensor& grad, Tensor& db);
void SubBackwardCUDA(const Tensor& grad, Tensor& db);

void MulBackwardCPU(const Tensor& a, const Tensor& b, const Tensor& grad,
                    Tensor& da, Tensor& db);
void MulBackwardCUDA(const Tensor& a, const Tensor& b, const Tensor& grad,
                     Tensor& da, Tensor& db);

void DivBackwardCPU(const Tensor& a, const Tensor& b, const Tensor& grad,
                    Tensor& da, Tensor& db);
void DivBackwardCUDA(const Tensor& a, const Tensor& b, const Tensor& grad,
                     Tensor& da, Tensor& db);

void NegBackwardCPU(const Tensor& grad, Tensor& dx);
void NegBackwardCUDA(const Tensor& grad, Tensor& dx);

void SqrtBackwardCPU(const Tensor& x, const Tensor& grad, Tensor& dx);
void SqrtBackwardCUDA(const Tensor& x, const Tensor& grad, Tensor& dx);

// 按各自 stride 读取 x 和上游梯度，写入连续 dx。
void RsqrtBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);
void RsqrtBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx);

void ExpBackwardCPU(const Tensor& y, const Tensor& grad, Tensor& dx);
void ExpBackwardCUDA(const Tensor& y, const Tensor& grad, Tensor& dx);

void LogBackwardCPU(const Tensor& x, const Tensor& grad, Tensor& dx);
void LogBackwardCUDA(const Tensor& x, const Tensor& grad, Tensor& dx);

}  // namespace mtensor
