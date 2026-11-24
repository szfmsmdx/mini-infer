#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/activation/kernels.h"
#include "mini_infer/core/tensor.h"
#include "ops/common/cuda_index.cuh"
#include "ops/common/cuda_unary.cuh"

namespace mtensor {
namespace {

template <typename In, typename Out>
__global__ void SigmoidKernel(const In* x, Out* out, std::size_t count, CUDALayout x_layout) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = GetNumelIndex(i, x_layout);
    const float v = static_cast<float>(x[xi]);
    const float t = expf(-fabsf(v));
    out[i] = static_cast<Out>(v >= 0.0f ? 1.0f / (1.0f + t) : t / (1.0f + t));
}

template <typename In, typename Out>
void LaunchSigmoid(const Tensor& x, Tensor& out) {
    LaunchUnary<In, Out, SigmoidKernel<In, Out>>(x, out);
}

template <typename In, typename Out>
__global__ void SiluKernel(const In* x, Out* out, std::size_t count, CUDALayout x_layout) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = GetNumelIndex(i, x_layout);
    const float v = static_cast<float>(x[xi]);
    float y;
    if (v < -80.0f) {
        // 负尾部用 double，避免 exp(x) 在乘 x 前下溢。
        const double t = exp(static_cast<double>(v));
        y = static_cast<float>(v * t / (1.0 + t));
    } else {
        const float t = expf(-fabsf(v));
        y = v >= 0.0f ? v / (1.0f + t) : (v * t) / (1.0f + t);
    }
    out[i] = static_cast<Out>(y);
}

template <typename In, typename Out>
void LaunchSilu(const Tensor& x, Tensor& out) {
    LaunchUnary<In, Out, SiluKernel<In, Out>>(x, out);
}

template <typename T>
__global__ void SigmoidBackwardKernel(const T* x, const T* grad, T* dx,
                                      std::size_t count, CUDALayout x_layout, CUDALayout grad_layout) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = GetNumelIndex(i, x_layout);
    const auto gi = GetNumelIndex(i, grad_layout);
    const float value = static_cast<float>(x[xi]);
    const float upstream = static_cast<float>(grad[gi]);
    const float t = expf(-fabsf(value));
    const float denominator = 1.0f + t;
    dx[i] = static_cast<T>((upstream * t) / (denominator * denominator));
}

template <typename T>
void LaunchSigmoidBackward(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    LaunchBackward<T, SigmoidBackwardKernel<T>>(x, output_grad, dx);
}

template <typename T>
__global__ void SiluBackwardKernel(const T* x, const T* grad, T* dx,
                                      std::size_t count, CUDALayout x_layout, CUDALayout grad_layout) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = GetNumelIndex(i, x_layout);
    const auto gi = GetNumelIndex(i, grad_layout);
    const float value = static_cast<float>(x[xi]);
    const float upstream = static_cast<float>(grad[gi]);
    const float s = 1.0f / (1.0f + expf(-value));
    dx[i] = static_cast<T>((upstream * s) * (1.0f + value * (1.0f - s)));
}

template <typename T>
void LaunchSiluBackward(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    LaunchBackward<T, SiluBackwardKernel<T>>(x, output_grad, dx);
}

}  // namespace

void SigmoidCUDA(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchSigmoid<float, float>(x, out); break;
        case Dtype::kFloat16: LaunchSigmoid<__half, __half>(x, out); break;
        case Dtype::kBFloat16: LaunchSigmoid<__nv_bfloat16, __nv_bfloat16>(x, out); break;
        case Dtype::kInt8: LaunchSigmoid<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: LaunchSigmoid<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: LaunchSigmoid<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: LaunchSigmoid<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: LaunchSigmoid<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("sigmoid: unsupported dtype");
    }
}

void SiluCUDA(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchSilu<float, float>(x, out); break;
        case Dtype::kFloat16: LaunchSilu<__half, __half>(x, out); break;
        case Dtype::kBFloat16: LaunchSilu<__nv_bfloat16, __nv_bfloat16>(x, out); break;
        case Dtype::kInt8: LaunchSilu<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: LaunchSilu<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: LaunchSilu<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: LaunchSilu<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: LaunchSilu<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("silu: unsupported dtype");
    }
}

void SigmoidBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchSigmoidBackward<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: LaunchSigmoidBackward<__half>(x, output_grad, dx); break;
        case Dtype::kBFloat16: LaunchSigmoidBackward<__nv_bfloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("sigmoid backward: unsupported dtype");
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
