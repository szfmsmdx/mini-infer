#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "autograd/backward_nodes/elementwise.h"
#include "ops/cuda/index.cuh"
#include "utils/memory.h"

namespace mtensor {
namespace {

template <typename T>
__global__ void SigmoidBackwardKernel(const T* x, const T* grad, T* dx,
                                      std::size_t count, const int* shape,
                                      const std::size_t* xs, const std::size_t* gs,
                                      int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = xs ? GetNumelIndex(i, shape, xs, ndim) : i;
    const auto gi = gs ? GetNumelIndex(i, shape, gs, ndim) : i;
    const float value = static_cast<float>(x[xi]);
    const float upstream = static_cast<float>(grad[gi]);
    const float t = expf(-fabsf(value));
    const float denominator = 1.0f + t;
    dx[i] = static_cast<T>((upstream * t) / (denominator * denominator));
}

template <typename T>
void LaunchSigmoidBackward(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto count = x.numel();
    if (count == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((count - 1) / threads + 1);
    SigmoidBackwardKernel<T><<<blocks, threads>>>(
        static_cast<const T*>(x.data_ptr()), static_cast<const T*>(output_grad.data_ptr()),
        static_cast<T*>(dx.data_ptr()), count, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(),
        output_grad.is_contiguous() ? nullptr : output_grad.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void RsqrtBackwardKernel(const T* x, const T* grad, T* dx,
                                      std::size_t count, const int* shape,
                                      const std::size_t* xs, const std::size_t* gs,
                                      int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = xs ? GetNumelIndex(i, shape, xs, ndim) : i;
    const auto gi = gs ? GetNumelIndex(i, shape, gs, ndim) : i;
    const float value = static_cast<float>(x[xi]);
    const float upstream = static_cast<float>(grad[gi]);
    const float y = rsqrtf(value);
    dx[i] = static_cast<T>(((-0.5f * y) * upstream) * y * y);
}

template <typename T>
void LaunchRsqrtBackward(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto count = x.numel();
    if (count == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((count - 1) / threads + 1);
    RsqrtBackwardKernel<T><<<blocks, threads>>>(
        static_cast<const T*>(x.data_ptr()), static_cast<const T*>(output_grad.data_ptr()),
        static_cast<T*>(dx.data_ptr()), count, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(),
        output_grad.is_contiguous() ? nullptr : output_grad.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void SiluBackwardKernel(const T* x, const T* grad, T* dx,
                                      std::size_t count, const int* shape,
                                      const std::size_t* xs, const std::size_t* gs,
                                      int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = xs ? GetNumelIndex(i, shape, xs, ndim) : i;
    const auto gi = gs ? GetNumelIndex(i, shape, gs, ndim) : i;
    const float value = static_cast<float>(x[xi]);
    const float upstream = static_cast<float>(grad[gi]);
    const float s = 1.0f / (1.0f + expf(-value));
    dx[i] = static_cast<T>((upstream * s) * (1.0f + value * (1.0f - s)));
}

template <typename T>
void LaunchSiluBackward(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto count = x.numel();
    if (count == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((count - 1) / threads + 1);
    SiluBackwardKernel<T><<<blocks, threads>>>(
        static_cast<const T*>(x.data_ptr()), static_cast<const T*>(output_grad.data_ptr()),
        static_cast<T*>(dx.data_ptr()), count, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(),
        output_grad.is_contiguous() ? nullptr : output_grad.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void SigmoidBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchSigmoidBackward<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: LaunchSigmoidBackward<__half>(x, output_grad, dx); break;
        case Dtype::kBFloat16: LaunchSigmoidBackward<__nv_bfloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("sigmoid backward: unsupported dtype");
    }
}

void RsqrtBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchRsqrtBackward<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: LaunchRsqrtBackward<__half>(x, output_grad, dx); break;
        case Dtype::kBFloat16: LaunchRsqrtBackward<__nv_bfloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("rsqrt backward: unsupported dtype");
    }
}

void SiluBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchSiluBackward<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: LaunchSiluBackward<__half>(x, output_grad, dx); break;
        case Dtype::kBFloat16: LaunchSiluBackward<__nv_bfloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("silu backward: unsupported dtype");
    }
}

}  // namespace mtensor
