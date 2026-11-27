#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/norm/kernels.h"
#include "backend/common/dtype_dispatch.h"
#include "mini_infer/tensor/tensor.h"
#include "backend/common/cuda_index.cuh"
#include "runtime/cuda_check.h"

namespace mtensor {
namespace {

// 每个线程处理一行，平方和及归一化中间量保持 float。
template <typename T>
__global__ void RMSNormKernel(
    const T* x, const T* weight, T* output,
    CUDALayout x_layout,
    std::size_t weight_stride, std::size_t rows, float eps
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const int features = x_layout.shape[x_layout.dim - 1];
    const std::size_t offset = GetNumelIndex(row * features, x_layout);
    const auto* x_row = x + offset;
    auto* output_row = output + row * features;

    float sum_squares = 0.0f;
    for (int j = 0; j < features; ++j) {
        const float x_value = static_cast<float>(x_row[j * x_layout.stride[x_layout.dim - 1]]);
        sum_squares += x_value * x_value;
    }
    const float mean_square = sum_squares / static_cast<float>(features);
    const float inv_rms = rsqrtf(mean_square + eps);

    for (int j = 0; j < features; ++j) {
        const float x_value = static_cast<float>(x_row[j * x_layout.stride[x_layout.dim - 1]]);
        const float weight_value = static_cast<float>(weight[j * weight_stride]);
        output_row[j] = static_cast<T>((x_value * inv_rms) * weight_value);
    }
}

template <typename T>
void LaunchRMSNorm(const Tensor& x, const Tensor& weight, float eps, Tensor& output) {
    const std::size_t rows = x.numel() / x.size(x.dim() - 1);
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((rows - 1) / threads + 1);
    const auto* input = static_cast<const T*>(x.data_ptr());
    const auto* scale = static_cast<const T*>(weight.data_ptr());
    auto* result = static_cast<T*>(output.data_ptr());
    RMSNormKernel<T><<<blocks, threads>>>(
        input, scale, result, PackLayout(x),
        weight.stride()[0], rows, eps);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void RMSNormCUDA(const Tensor& x, const Tensor& weight, float eps, Tensor& output) {
    DispatchFloatingDtypes(x.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, float>) LaunchRMSNorm<float>(x, weight, eps, output);
        else if constexpr (std::is_same_v<T, Float16>) LaunchRMSNorm<__half>(x, weight, eps, output);
        else LaunchRMSNorm<__nv_bfloat16>(x, weight, eps, output);
    });
}

namespace {

// 每个线程处理一行，计算输入梯度并保存 FP32 的 inv_rms。
template <typename T>
__global__ void RMSNormInputBackwardKernel(
    const T* x, const T* weight, const T* grad, T* dx, float* row_inv_rms,
    CUDALayout x_layout, CUDALayout grad_layout,
    std::size_t weight_stride, std::size_t rows, float eps
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const int features = x_layout.shape[x_layout.dim - 1];
    const auto* x_row = x + GetNumelIndex(row * features, x_layout);
    const auto* grad_row = grad + GetNumelIndex(row * features, grad_layout);
    auto* dx_row = dx + row * features;

    float sum_squares = 0.0f;
    float sum_grad_weight_x = 0.0f;
    for (int j = 0; j < features; ++j) {
        const float x_value = static_cast<float>(x_row[j * x_layout.stride[x_layout.dim - 1]]);
        const float grad_value = static_cast<float>(grad_row[j * grad_layout.stride[grad_layout.dim - 1]]);
        const float weight_value = static_cast<float>(weight[j * weight_stride]);
        sum_squares += x_value * x_value;
        sum_grad_weight_x += grad_value * weight_value * x_value;
    }
    const float mean_square = sum_squares / static_cast<float>(features);
    const float inv_rms = rsqrtf(mean_square + eps);
    row_inv_rms[row] = inv_rms;
    // 用归一化输入计算梯度，避免先算 RMS 倒数的三次方。
    const float mean_grad_weight_x_hat =
        (sum_grad_weight_x / static_cast<float>(features)) * inv_rms;

    for (int j = 0; j < features; ++j) {
        const float x_value = static_cast<float>(x_row[j * x_layout.stride[x_layout.dim - 1]]);
        const float grad_value = static_cast<float>(grad_row[j * grad_layout.stride[grad_layout.dim - 1]]);
        const float weight_value = static_cast<float>(weight[j * weight_stride]);
        const float x_hat = x_value * inv_rms;
        const float weighted_grad = grad_value * weight_value;
        const float dx_value = (weighted_grad - x_hat * mean_grad_weight_x_hat) * inv_rms;
        dx_row[j] = static_cast<T>(dx_value);
    }
}

// 每个线程负责一个特征，权重梯度跨所有行以 float 累加，最后写回。
template <typename T>
__global__ void RMSNormWeightBackwardKernel(
    const T* x, const T* grad, const float* row_inv_rms, T* dw,
    CUDALayout x_layout, CUDALayout grad_layout,
    std::size_t rows
) {
    const std::size_t j =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int features = x_layout.shape[x_layout.dim - 1];
    if (j >= features) return;

    float weight_grad = 0.0f;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t index = row * features + j;
        const std::size_t x_offset = GetNumelIndex(index, x_layout);
        const std::size_t grad_offset = GetNumelIndex(index, grad_layout);
        const float x_value = static_cast<float>(x[x_offset]);
        const float grad_value = static_cast<float>(grad[grad_offset]);
        weight_grad += grad_value * x_value * row_inv_rms[row];
    }
    dw[j] = static_cast<T>(weight_grad);
}

template <typename T>
void LaunchRMSNormBackward(const Tensor& x, const Tensor& weight, const Tensor& output_grad,
                           float eps, Tensor& dx, Tensor& dw) {
    const int features = x.size(x.dim() - 1);
    const std::size_t rows = x.numel() / features;
    auto row_shape = x.shape().to_vector();
    row_shape.back() = 1;
    Tensor row_inv_rms(row_shape, Dtype::kFloat32, x.device());
    constexpr unsigned int threads = 256;
    const auto row_blocks = static_cast<unsigned int>((rows - 1) / threads + 1);
    const auto feature_blocks = static_cast<unsigned int>((features - 1) / threads + 1);

    const auto* input = static_cast<const T*>(x.data_ptr());
    const auto* scale = static_cast<const T*>(weight.data_ptr());
    const auto* grad = static_cast<const T*>(output_grad.data_ptr());
    auto* input_grad = static_cast<T*>(dx.data_ptr());
    auto* inv_rms = static_cast<float*>(row_inv_rms.data_ptr());
    RMSNormInputBackwardKernel<T><<<row_blocks, threads>>>(
        input, scale, grad, input_grad, inv_rms, PackLayout(x),
        PackLayout(output_grad), weight.stride()[0], rows, eps);
    CUDA_CHECK(cudaGetLastError());

    // 同一 stream 按序执行，第二个 kernel 读取第一个 kernel 写入的 inv_rms。
    auto* weight_grad = static_cast<T*>(dw.data_ptr());
    RMSNormWeightBackwardKernel<T><<<feature_blocks, threads>>>(
        input, grad, inv_rms, weight_grad, PackLayout(x),
        PackLayout(output_grad), rows);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void RMSNormBackwardCUDA(const Tensor& x, const Tensor& weight, const Tensor& output_grad,
                         float eps, Tensor& dx, Tensor& dw) {
    DispatchFloatingDtypes(x.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, float>)
            LaunchRMSNormBackward<float>(x, weight, output_grad, eps, dx, dw);
        else if constexpr (std::is_same_v<T, Float16>)
            LaunchRMSNormBackward<__half>(x, weight, output_grad, eps, dx, dw);
        else
            LaunchRMSNormBackward<__nv_bfloat16>(x, weight, output_grad, eps, dx, dw);
    });
}

}  // namespace mtensor
