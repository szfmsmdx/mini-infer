#pragma once

#include <cuda_runtime.h>

#include "ops/common/cuda_layout.h"
#include "mini_infer/core/tensor.h"
#include "device/memory.h"

namespace mtensor {

template <typename In, typename Out, auto Kernel>
void LaunchUnary(const Tensor& x, Tensor& out) {
    const auto n = x.numel();
    if (n == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n - 1) / threads + 1);
    Kernel<<<blocks, threads>>>(
        static_cast<const In*>(x.data_ptr()), static_cast<Out*>(out.data_ptr()), n, PackLayout(x));
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
        input, grad, input_grad, count, PackLayout(x), PackLayout(output_grad));
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
