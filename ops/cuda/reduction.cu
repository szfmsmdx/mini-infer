#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/backend.h"
#include "ops/cuda/index.cuh"
#include "utils/memory.h"

namespace mtensor {
namespace {

constexpr int kThreads = 1024;
constexpr int kWarpSize = 32;

// 每个 warp 先归约自己的值，再由第一个 warp 归约 32 个部分和。
template <typename T>
__device__ T ReduceBlock(T value, T* shared) {
    int tid = threadIdx.x;
    int warp = tid / kWarpSize;
    int lane = tid % kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = shared[lane];
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;
}

__device__ float ReadFloat(const void* input, Dtype dtype, std::size_t offset) {
    switch (dtype) {
        case Dtype::kFloat32: return static_cast<const float*>(input)[offset];
        case Dtype::kFloat16: return __half2float(static_cast<const __half*>(input)[offset]);
        case Dtype::kBFloat16: return __bfloat162float(static_cast<const __nv_bfloat16*>(input)[offset]);
        default: return 0.0f;
    }
}

__device__ unsigned long long ReadInteger(const void* input, Dtype dtype, std::size_t offset) {
    switch (dtype) {
        case Dtype::kInt8: return static_cast<unsigned long long>(static_cast<const std::int8_t*>(input)[offset]);
        case Dtype::kUInt8: return static_cast<const std::uint8_t*>(input)[offset];
        case Dtype::kInt16: return static_cast<unsigned long long>(static_cast<const std::int16_t*>(input)[offset]);
        case Dtype::kInt32: return static_cast<unsigned long long>(static_cast<const std::int32_t*>(input)[offset]);
        case Dtype::kInt64: return static_cast<unsigned long long>(static_cast<const std::int64_t*>(input)[offset]);
        default: return 0;
    }
}

// 每个 warp 负责一段输入，warp 内的线程以 32 为步长累加。
template <typename T>
__global__ void SumKernel(const void* input, void* output, Dtype dtype,
                          const int* shape, const std::size_t* stride,
                          int ndim, std::size_t numel) {
    __shared__ T shared[kWarpSize];
    int idx = threadIdx.x;
    int warp = idx / kWarpSize;
    int lane = idx % kWarpSize;
    const std::size_t warp_elements = ((numel - 1) / kThreads + 1) * kWarpSize;
    const std::size_t warp_start = warp * warp_elements;
    const std::size_t warp_end = warp_start + warp_elements;
    T value = 0;
    for (std::size_t i = warp_start + lane; i < warp_end && i < numel; i += kWarpSize) {
        const std::size_t offset = GetNumelIndex(i, shape, stride, ndim);
        if constexpr (std::is_same_v<T, float>) {
            value += ReadFloat(input, dtype, offset);
        } else {
            value += ReadInteger(input, dtype, offset);
        }
    }

    T total = ReduceBlock(value, shared);
    if (idx == 0) {
        if constexpr (std::is_same_v<T, float>) {
            if (dtype == Dtype::kFloat32) {
                *static_cast<float*>(output) = total;
            } else if (dtype == Dtype::kFloat16) {
                *static_cast<__half*>(output) = __float2half_rn(total);
            } else {
                *static_cast<__nv_bfloat16*>(output) = __float2bfloat16_rn(total);
            }
        } else {
            // 按位保存 64 位回绕结果，与 CPU 整数归约一致。
            memcpy(output, &total, sizeof(total));
        }
    }
}

}  // namespace

void SumCUDA(const void* input, void* output, Dtype dtype, const int* shape,
             const std::size_t* stride, int ndim, std::size_t numel) {
    const bool floating = dtype == Dtype::kFloat32 || dtype == Dtype::kFloat16 ||
                          dtype == Dtype::kBFloat16;
    if (floating) {
        SumKernel<float><<<1, kThreads>>>(input, output, dtype, shape, stride, ndim, numel);
    } else {
        SumKernel<unsigned long long><<<1, kThreads>>>(
            input, output, dtype, shape, stride, ndim, numel);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
