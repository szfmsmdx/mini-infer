#include <cstddef>
#include <cuda_runtime.h>

#include "mini_infer/ops/position_embedding.h"
#include "cuda/cuda_check.h"

namespace mtensor {
namespace {

constexpr unsigned int kThreads = 128;

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>((count + kThreads - 1) / kThreads);
}

__global__ void RoPEKernel(const float* x, int width, int head_width, float base, bool inverse,
                           std::size_t pairs, float* output) {
    const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    const std::size_t offset = pair * 2;
    const float position = static_cast<float>(offset / width);
    const int col = static_cast<int>(offset % width);
    const float exponent = -static_cast<float>(col % head_width) / head_width;
    const float angle = position * powf(base, exponent);
    float sine;
    float cosine;
    sincosf(inverse ? -angle : angle, &sine, &cosine);
    const float first = x[offset];
    const float second = x[offset + 1];
    output[offset] = first * cosine - second * sine;
    output[offset + 1] = first * sine + second * cosine;
}

}  // namespace

void RoPECUDA(const Tensor& x, int head_width, float base, bool inverse, Tensor& output) {
    const std::size_t pairs = x.numel() / 2;
    const auto* input = static_cast<const float*>(x.data_ptr());
    auto* result = static_cast<float*>(output.data_ptr());
    RoPEKernel<<<Blocks(pairs), kThreads>>>(
        input, x.size(1), head_width, base, inverse, pairs, result);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
