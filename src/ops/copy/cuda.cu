#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ops/copy/kernels.h"
#include "mini_infer/core/tensor.h"
#include "ops/common/cuda_index.cuh"
#include "device/memory.h"

namespace mtensor {

namespace {

__device__ float ReadFloat(const void* data, Dtype dtype, std::size_t offset) {
    switch (dtype) {
        case Dtype::kFloat16: return __half2float(static_cast<const __half*>(data)[offset]);
        case Dtype::kBFloat16: return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[offset]);
        case Dtype::kInt8: return static_cast<const std::int8_t*>(data)[offset];
        case Dtype::kUInt8: return static_cast<const std::uint8_t*>(data)[offset];
        case Dtype::kInt16: return static_cast<const std::int16_t*>(data)[offset];
        case Dtype::kInt32: return static_cast<float>(static_cast<const std::int32_t*>(data)[offset]);
        case Dtype::kInt64: return static_cast<float>(static_cast<const std::int64_t*>(data)[offset]);
        default: return 0.0f;
    }
}

__global__ void CastKernel(const void* input, void* output, Dtype in_dtype, Dtype out_dtype,
                           CUDALayout layout, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::size_t offset = GetNumelIndex(i, layout);
    if (out_dtype == Dtype::kFloat32) {
        static_cast<float*>(output)[i] = ReadFloat(input, in_dtype, offset);
    } else {
        const float value = static_cast<const float*>(input)[offset];
        if (out_dtype == Dtype::kFloat16) static_cast<__half*>(output)[i] = __float2half_rn(value);
        else static_cast<__nv_bfloat16*>(output)[i] = __float2bfloat16_rn(value);
    }
}

}  // namespace

__global__ void CopyKernel(std::byte* dst, const std::byte* src, CUDALayout dst_layout, CUDALayout src_layout, std::size_t numel, std::size_t element_size) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        const auto dst_offset = GetNumelIndex(idx, dst_layout) * element_size;
        const auto src_offset = GetNumelIndex(idx, src_layout) * element_size;
        memcpy(dst + dst_offset, src + src_offset, element_size);
    }
}

void CopyCUDA(void* destination, const void* source, CUDALayout dst_layout, CUDALayout src_layout, std::size_t numel, std::size_t element_size) {
    if (numel == 0) return;
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((numel - 1) / threads + 1);
    auto* dst = static_cast<std::byte*>(destination);
    const auto* src = static_cast<const std::byte*>(source);
    CopyKernel<<<blocks, threads>>>(
        dst, src, dst_layout, src_layout, numel, element_size);
    CUDA_CHECK(cudaGetLastError());
}

void CastCUDA(const Tensor& input, Tensor& output) {
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((input.numel() - 1) / threads + 1);
    CastKernel<<<blocks, threads>>>(
        input.data_ptr(), output.data_ptr(), input.dtype(), output.dtype(),
        PackLayout(input), input.numel());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
