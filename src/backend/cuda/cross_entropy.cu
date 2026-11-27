#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "mini_infer/ops/loss_fn.h"
#include "runtime/cuda_check.h"

namespace mtensor {
namespace {

constexpr unsigned int kThreads = 128;

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>((count + kThreads - 1) / kThreads);
}

__device__ float Maximum(const float* row, int width) {
    float maximum = row[0];
    for (int i = 1; i < width; ++i) {
        maximum = fmaxf(maximum, row[i]);
    }
    return maximum;
}

__device__ float SumShiftedExp(const float* row, int width, float maximum) {
    float sum = 0.0f;
    for (int i = 0; i < width; ++i) {
        sum += expf(row[i] - maximum);
    }
    return sum;
}

__global__ void CrossEntropyKernel(const float* logits, const std::int32_t* ids,
                                   int length, int width, float* losses) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= length) return;
    const float* row = logits + i * width;
    const float maximum = Maximum(row, width);
    losses[i] = (maximum - row[ids[i]]) + logf(SumShiftedExp(row, width, maximum));
}

__global__ void CrossEntropyGradKernel(const float* logits, const std::int32_t* ids,
                                       const float* grad, int length, int width, float* result) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= length) return;
    const std::size_t offset = i * width;
    const float* row = logits + offset;
    const float maximum = Maximum(row, width);
    const float sum = SumShiftedExp(row, width, maximum);
    const float scale = *grad / length;
    for (int col = 0; col < width; ++col) {
        const float p = expf(row[col] - maximum) / sum;
        result[offset + col] = (p - static_cast<float>(col == ids[i])) * scale;
    }
}

}  // namespace

void CrossEntropyCUDA(const Tensor& logits, const Tensor& targets, Tensor& row_losses) {
    const auto* scores = static_cast<const float*>(logits.data_ptr());
    const auto* ids = static_cast<const std::int32_t*>(targets.data_ptr());
    auto* losses = static_cast<float*>(row_losses.data_ptr());
    CrossEntropyKernel<<<Blocks(logits.size(0)), kThreads>>>(
        scores, ids, logits.size(0), logits.size(1), losses);
    CUDA_CHECK(cudaGetLastError());
}

void CrossEntropyGradCUDA(const Tensor& logits, const Tensor& targets,
                          const Tensor& grad, Tensor& logits_grad) {
    const auto* scores = static_cast<const float*>(logits.data_ptr());
    const auto* ids = static_cast<const std::int32_t*>(targets.data_ptr());
    const auto* upstream = static_cast<const float*>(grad.data_ptr());
    auto* scores_grad = static_cast<float*>(logits_grad.data_ptr());
    CrossEntropyGradKernel<<<Blocks(logits.size(0)), kThreads>>>(
        scores, ids, upstream, logits.size(0), logits.size(1), scores_grad);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
