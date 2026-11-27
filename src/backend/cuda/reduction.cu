#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mini_infer/tensor/tensor.h"
#include "ops/reduction/kernels.h"
#include "backend/common/cuda_index.cuh"
#include "runtime/cuda_check.h"

namespace mtensor {
namespace {

constexpr int kThreads = 1024;
constexpr int kWarpSize = 32;

// 每个 warp 先归约自己的值，再由第一个 warp 归约 32 个部分和。
template <typename T>
__device__ T BlockReduceSum(T value, T* shared) {
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
                          CUDALayout layout, std::size_t numel) {
    __shared__ T shared[kWarpSize];
    int idx = threadIdx.x;
    int warp = idx / kWarpSize;
    int lane = idx % kWarpSize;
    const std::size_t warp_elements = ((numel - 1) / kThreads + 1) * kWarpSize;
    const std::size_t warp_start = warp * warp_elements;
    const std::size_t warp_end = warp_start + warp_elements;

    // 单个线程 sum
    T value = 0;
    for (std::size_t i = warp_start + lane; i < warp_end && i < numel; i += kWarpSize) {
        const std::size_t offset = GetNumelIndex(i, layout);
        if constexpr (std::is_same_v<T, float>) {
            value += ReadFloat(input, dtype, offset);
        } else {
            value += ReadInteger(input, dtype, offset);
        }
    }

    // 把各线程的局部和归约成块内总和。
    T total = BlockReduceSum(value, shared);
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

// 每个线程负责一个输出元素；相邻线程沿输出逻辑顺序读取输入。
template <typename T>
__global__ void SumKernel(const void* input, void* output, Dtype dtype,
                             CUDALayout layout, std::size_t output_numel, int dim) {
    const std::size_t out = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out >= output_numel) return;
    std::size_t index = out;
    std::size_t base = 0;
    for (int axis = layout.dim - 1; axis >= 0; --axis) {
        if (axis == dim) continue;
        base += (index % layout.shape[axis]) * layout.stride[axis];
        index /= layout.shape[axis];
    }

    // 单线程 sum
    T value = 0;
    for (int j = 0; j < layout.shape[dim]; ++j) {
        const std::size_t offset = base + j * layout.stride[dim];
        if constexpr (std::is_same_v<T, float>) value += ReadFloat(input, dtype, offset);
        else value += ReadInteger(input, dtype, offset);
    }
    
    if constexpr (std::is_same_v<T, float>) {
        if (dtype == Dtype::kFloat32) static_cast<float*>(output)[out] = value;
        else if (dtype == Dtype::kFloat16)
            static_cast<__half*>(output)[out] = __float2half_rn(value);
        else static_cast<__nv_bfloat16*>(output)[out] = __float2bfloat16_rn(value);
    } else {
        memcpy(static_cast<std::byte*>(output) + out * sizeof(value), &value, sizeof(value));
    }
}

constexpr int kExtremaThreads = 256;

template <typename T>
__device__ T ReadExtrema(const void* input, Dtype dtype, std::size_t offset) {
    if constexpr (std::is_same_v<T, float>) {
        return ReadFloat(input, dtype, offset);
    } else {
        switch (dtype) {
            case Dtype::kInt8: return static_cast<const std::int8_t*>(input)[offset];
            case Dtype::kUInt8: return static_cast<const std::uint8_t*>(input)[offset];
            case Dtype::kInt16: return static_cast<const std::int16_t*>(input)[offset];
            case Dtype::kInt32: return static_cast<const std::int32_t*>(input)[offset];
            case Dtype::kInt64: return static_cast<const std::int64_t*>(input)[offset];
            default: return 0;
        }
    }
}

template <typename T>
__device__ bool PreferExtrema(T value, std::size_t index, T best, std::size_t best_index, bool maximum) {
    if constexpr (std::is_same_v<T, float>) {
        const bool nan = isnan(value);
        const bool best_nan = isnan(best);
        if (nan != best_nan) return nan;
        if (nan) return index < best_index;
    }
    if (value == best) return index < best_index;
    return maximum ? value > best : value < best;
}

__device__ std::size_t GroupStart(std::size_t group, std::size_t count, std::size_t inner_size) {
    return group / inner_size * count * inner_size + group % inner_size;
}

// 一个 block 合作归约一组，等值时保留最早元素的原始编码。
template <typename T>
__global__ void ExtremaKernel(const void* input, void* output, Dtype dtype,
                              CUDALayout layout,
                              std::size_t groups, std::size_t count, std::size_t inner_size,
                              std::size_t element_size, bool maximum) {
    __shared__ T best_values[kExtremaThreads];
    __shared__ std::size_t best_indices[kExtremaThreads];
    const unsigned int tid = threadIdx.x;
    for (std::size_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const std::size_t start = GroupStart(group, count, inner_size);
        std::size_t best = count;
        T best_value{};
        for (std::size_t j = tid; j < count; j += kExtremaThreads) {
            const std::size_t i = start + j * inner_size;
            const std::size_t offset = GetNumelIndex(i, layout);
            const T value = ReadExtrema<T>(input, dtype, offset);
            if (best == count || PreferExtrema(value, j, best_value, best, maximum)) {
                best = j;
                best_value = value;
            }
        }
        best_values[tid] = best_value;
        best_indices[tid] = best;
        __syncthreads();
        for (unsigned int step = kExtremaThreads / 2; step > 0; step /= 2) {
            if (tid < step) {
                const auto other = best_indices[tid + step];
                const auto value = best_values[tid + step];
                if (other != count && (best_indices[tid] == count ||
                    PreferExtrema(value, other, best_values[tid], best_indices[tid], maximum))) {
                    best_indices[tid] = other;
                    best_values[tid] = value;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const std::size_t i = start + best_indices[0] * inner_size;
            const std::size_t offset = GetNumelIndex(i, layout);
            const auto* bytes = static_cast<const std::byte*>(input);
            auto* result = static_cast<std::byte*>(output);
            memcpy(result + group * element_size, bytes + offset * element_size, element_size);
        }
        __syncthreads();
    }
}

__device__ bool MatchesExtrema(const void* input, Dtype dtype, std::size_t i, float best,
                                const CUDALayout& layout) {
    const float value = ReadFloat(input, dtype, GetNumelIndex(i, layout));
    return value == best || (isnan(value) && isnan(best));
}

__device__ void WriteFloat(void* output, Dtype dtype, std::size_t i, float value) {
    if (dtype == Dtype::kFloat32) static_cast<float*>(output)[i] = value;
    else if (dtype == Dtype::kFloat16) static_cast<__half*>(output)[i] = __float2half_rn(value);
    else static_cast<__nv_bfloat16*>(output)[i] = __float2bfloat16_rn(value);
}

__global__ void ExtremaGradKernel(const void* input, const void* extrema, const void* grad, void* input_grad,
                                  Dtype dtype, CUDALayout layout, std::size_t groups, std::size_t count,
                                  std::size_t inner_size) {
    __shared__ std::size_t tie_counts[kExtremaThreads];
    const unsigned int tid = threadIdx.x;
    for (std::size_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const std::size_t start = GroupStart(group, count, inner_size);
        const float best = ReadFloat(extrema, dtype, group);
        std::size_t ties = 0;
        for (std::size_t j = tid; j < count; j += kExtremaThreads) {
            ties += MatchesExtrema(input, dtype, start + j * inner_size, best, layout);
        }
        tie_counts[tid] = ties;
        __syncthreads();
        for (unsigned int step = kExtremaThreads / 2; step > 0; step /= 2) {
            if (tid < step) tie_counts[tid] += tie_counts[tid + step];
            __syncthreads();
        }
        const float share = ReadFloat(grad, dtype, group) / static_cast<float>(tie_counts[0]);
        for (std::size_t j = tid; j < count; j += kExtremaThreads) {
            const std::size_t i = start + j * inner_size;
            const bool matches = MatchesExtrema(input, dtype, i, best, layout);
            WriteFloat(input_grad, dtype, i, matches ? share : 0.0f);
        }
        __syncthreads();
    }
}

}  // namespace

void SumCUDA(const void* input, void* output, Dtype dtype, CUDALayout layout, std::size_t numel) {
    const bool floating = dtype == Dtype::kFloat32 || dtype == Dtype::kFloat16 ||
                          dtype == Dtype::kBFloat16;
    if (floating) {
        SumKernel<float><<<1, kThreads>>>(input, output, dtype, layout, numel);
    } else {
        SumKernel<unsigned long long><<<1, kThreads>>>(input, output, dtype, layout, numel);
    }
    CUDA_CHECK(cudaGetLastError());
}

void SumCUDA(const void* input, void* output, Dtype dtype, CUDALayout layout, std::size_t output_numel, int dim) {
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((output_numel + threads - 1) / threads);
    const bool floating = dtype == Dtype::kFloat32 || dtype == Dtype::kFloat16 ||
                          dtype == Dtype::kBFloat16;
    if (floating) {
        SumKernel<float><<<blocks, threads>>>(
            input, output, dtype, layout, output_numel, dim);
    } else {
        SumKernel<unsigned long long><<<blocks, threads>>>(
            input, output, dtype, layout, output_numel, dim);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ExtremaCUDA(const Tensor& input, Tensor& output, std::size_t inner_size, bool maximum) {
    const Dtype dtype = input.dtype();
    const std::size_t groups = output.numel();
    const std::size_t count = input.numel() / groups;
    const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(groups, 65535));
    if (dtype == Dtype::kFloat32 || dtype == Dtype::kFloat16 || dtype == Dtype::kBFloat16) {
        ExtremaKernel<float><<<blocks, kExtremaThreads>>>(
            input.data_ptr(), output.data_ptr(), dtype, PackLayout(input),
            groups, count, inner_size, input.element_size(), maximum);
    } else {
        ExtremaKernel<std::int64_t><<<blocks, kExtremaThreads>>>(
            input.data_ptr(), output.data_ptr(), dtype, PackLayout(input),
            groups, count, inner_size, input.element_size(), maximum);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ExtremaGradCUDA(const Tensor& input, const Tensor& extrema, const Tensor& grad,
                     Tensor& input_grad, std::size_t inner_size) {
    const std::size_t groups = extrema.numel();
    const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(groups, 65535));
    ExtremaGradKernel<<<blocks, kExtremaThreads>>>(
        input.data_ptr(), extrema.data_ptr(), grad.data_ptr(), input_grad.data_ptr(), input.dtype(),
        PackLayout(input), groups,
        input.numel() / groups, inner_size);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
