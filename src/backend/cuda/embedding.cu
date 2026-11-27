#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "mini_infer/ops/embedding.h"
#include "runtime/cuda_check.h"

namespace mtensor {
namespace {

constexpr unsigned int kThreads = 128;

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>((count + kThreads - 1) / kThreads);
}

__global__ void EmbeddingKernel(const float* weight, const std::int32_t* ids,
                                int width, std::size_t count, float* output) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const std::size_t index = static_cast<std::size_t>(ids[i / width]) * width + i % width;
        output[i] = weight[index];
    }
}

__global__ void EmbeddingGradKernel(const float* grad, const std::int32_t* ids,
                                    int width, std::size_t count, float* weight_grad) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const std::size_t index = static_cast<std::size_t>(ids[i / width]) * width + i % width;
        atomicAdd(weight_grad + index, grad[i]);
    }
}

}  // namespace

void EmbeddingCUDA(const Tensor& weight, const Tensor& indices, Tensor& output) {
    const auto* weights = static_cast<const float*>(weight.data_ptr());
    const auto* ids = static_cast<const std::int32_t*>(indices.data_ptr());
    auto* result = static_cast<float*>(output.data_ptr());
    EmbeddingKernel<<<Blocks(output.numel()), kThreads>>>(
        weights, ids, weight.size(1), output.numel(), result);
    CUDA_CHECK(cudaGetLastError());
}

void EmbeddingGradCUDA(const Tensor& grad, const Tensor& indices, Tensor& weight_grad) {
    const auto* upstream = static_cast<const float*>(grad.data_ptr());
    const auto* ids = static_cast<const std::int32_t*>(indices.data_ptr());
    auto* weight_grads = static_cast<float*>(weight_grad.data_ptr());
    EmbeddingGradKernel<<<Blocks(grad.numel()), kThreads>>>(
        upstream, ids, weight_grad.size(1), grad.numel(), weight_grads);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
