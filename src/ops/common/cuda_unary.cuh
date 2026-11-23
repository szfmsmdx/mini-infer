#pragma once

#include <cuda_runtime.h>

#include "mini_infer/tensor.h"
#include "device/memory.h"

namespace mtensor {

template <typename In, typename Out, auto Kernel>
void LaunchUnary(const Tensor& x, Tensor& out) {
    const auto n = x.numel();
    if (n == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n - 1) / threads + 1);
    Kernel<<<blocks, threads>>>(
        static_cast<const In*>(x.data_ptr()), static_cast<Out*>(out.data_ptr()), n, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, auto Kernel>
void LaunchBackward(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto count = x.numel();
    if (count == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((count - 1) / threads + 1);
    const auto* input = static_cast<const T*>(x.data_ptr());
    const auto* grad = static_cast<const T*>(output_grad.data_ptr());
    auto* input_grad = static_cast<T*>(dx.data_ptr());
    Kernel<<<blocks, threads>>>(
        input, grad, input_grad, count, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(),
        output_grad.is_contiguous() ? nullptr : output_grad.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
