#include <cstddef>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "autograd/backend.h"
#include "mini_infer/tensor.h"
#include "ops/cuda/index.cuh"
#include "utils/memory.h"

namespace mtensor {
namespace {

// 连续输入直接用 i；视图按各自的 stride 定位。输出总是连续。
template <typename T>
__global__ void RsqrtBackwardKernel(const T* x, T* out, std::size_t count, const int* shape,
                            const std::size_t* xs, int ndim, const T* grad, const std::size_t* gs) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = xs ? GetNumelIndex(i, shape, xs, ndim) : i;
    const float y = rsqrtf(static_cast<float>(x[xi]));
    const auto gi = gs ? GetNumelIndex(i, shape, gs, ndim) : i;
    out[i] = static_cast<T>((-0.5f * y * static_cast<float>(grad[gi])) * y * y);
}

template <typename T>
void LaunchRsqrtBackward(const Tensor& x, const Tensor& grad, Tensor& dx) {
    const auto n = x.numel();
    if (n == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n - 1) / threads + 1);
    RsqrtBackwardKernel<T><<<blocks, threads>>>(
        static_cast<const T*>(x.data_ptr()), static_cast<T*>(dx.data_ptr()), n, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(), x.dim(),
        static_cast<const T*>(grad.data_ptr()),
        grad.is_contiguous() ? nullptr : grad.stride_cuda());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void RsqrtBackwardCUDA(const Tensor& x, const Tensor& grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchRsqrtBackward<float>(x, grad, dx); break;
        case Dtype::kFloat16: LaunchRsqrtBackward<__half>(x, grad, dx); break;
        case Dtype::kBFloat16: LaunchRsqrtBackward<__nv_bfloat16>(x, grad, dx); break;
        default: throw std::invalid_argument("rsqrt backward: unsupported dtype");
    }
}

}  // namespace mtensor
