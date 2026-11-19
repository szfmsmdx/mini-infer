#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/backend.h"
#include "ops/cuda/index.cuh"
#include "utils/memory.h"

namespace mtensor {
namespace {

template <typename Input, typename Output>
__global__ void SqrtKernel(const Input* input, Output* output, std::size_t numel,
                           const int* shape, const std::size_t* stride, int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float x = static_cast<float>(input[GetNumelIndex(i, shape, stride, ndim)]);
    output[i] = static_cast<Output>(sqrtf(x));
}

template <typename Input, typename Output>
void LaunchSqrt(const void* input, void* output, std::size_t numel,
                const int* shape, const std::size_t* stride, int ndim) {
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((numel - 1) / threads + 1);
    SqrtKernel<<<blocks, threads>>>(static_cast<const Input*>(input), static_cast<Output*>(output),
                                    numel, shape, stride, ndim);
}

}  // namespace

void SqrtCUDA(const void* input, void* output, Dtype dtype, std::size_t numel,
              const int* shape, const std::size_t* stride, int ndim) {
    if (numel == 0) return;
    switch (dtype) {
        case Dtype::kFloat32: LaunchSqrt<float, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kFloat16: LaunchSqrt<__half, __half>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kBFloat16: LaunchSqrt<__nv_bfloat16, __nv_bfloat16>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kInt8: LaunchSqrt<std::int8_t, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kUInt8: LaunchSqrt<std::uint8_t, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kInt16: LaunchSqrt<std::int16_t, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kInt32: LaunchSqrt<std::int32_t, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kInt64: LaunchSqrt<std::int64_t, float>(input, output, numel, shape, stride, ndim); break;
        default: throw std::invalid_argument("sqrt: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
