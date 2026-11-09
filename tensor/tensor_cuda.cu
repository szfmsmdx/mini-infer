#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "tensor/tensor.h"
#include "utils/memory.h"

namespace mtensor{

__device__ std::size_t GetNumelIndex(
    std::size_t idx, const int* shape_cuda, const std::size_t* stride_cuda, int ndim
){
    std::size_t offset = 0;
    for(int d = ndim - 1; d >= 0; d--){
        std::size_t cur_offset = idx % shape_cuda[d];
        idx /= shape_cuda[d];
        offset += cur_offset * stride_cuda[d];
    }
    return offset;
}

template <typename Element>
__global__ void FillKernel(Element* device_ptr, int numel, Element value,
                           const int* shape, const std::size_t* stride, int ndim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numel){
        device_ptr[GetNumelIndex(idx, shape, stride, ndim)] = value;
    }
}

template <typename Element>
void LaunchTypedFill(
    void* device_ptr,
    std::size_t numel,
    const void* host_value,
    unsigned int blocks,
    unsigned int threads,
    const int* shape, const std::size_t* stride, int ndim
){
    const Element value = *static_cast<const Element*>(host_value); // 不修改原 host_value
    FillKernel<Element><<<blocks, threads>>>(
        static_cast<Element*>(device_ptr), numel, value, shape, stride, ndim);
}

void FillCUDA(
    void* device_ptr,
    std::size_t numel,
    Dtype dtype,
    const void* host_value,
    const int* shape, const std::size_t* stride, int ndim
){
    if(numel == 0){
        return;
    }

    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((numel - 1) / threads + 1);
    switch (dtype) {
        case Dtype::kFloat16:
        case Dtype::kBFloat16: LaunchTypedFill<std::uint16_t>(device_ptr, numel, host_value, blocks, threads, shape, stride, ndim); break;
        case Dtype::kFloat32: LaunchTypedFill<float>(device_ptr, numel, host_value, blocks, threads, shape, stride, ndim); break;
        case Dtype::kInt8: LaunchTypedFill<std::int8_t>(device_ptr, numel, host_value, blocks, threads, shape, stride, ndim); break;
        case Dtype::kUInt8: LaunchTypedFill<std::uint8_t>(device_ptr, numel, host_value, blocks, threads, shape, stride, ndim); break;
        case Dtype::kInt16: LaunchTypedFill<std::int16_t>(device_ptr, numel, host_value, blocks, threads, shape, stride, ndim); break;
        case Dtype::kInt32: LaunchTypedFill<std::int32_t>(device_ptr, numel, host_value, blocks, threads, shape, stride, ndim); break;
        case Dtype::kInt64: LaunchTypedFill<std::int64_t>(device_ptr, numel, host_value, blocks, threads, shape, stride, ndim); break;
        default: throw std::invalid_argument("unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}

__global__ void ZeroKernel(
    void* data_ptr,
    const int* shape_cuda,
    const std::size_t* stride_cuda,
    int ndim,
    std::size_t element_size,
    int numel
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < numel){
        auto* byted_data_ptr = static_cast<std::byte*>(data_ptr);
        std::size_t numel_index = GetNumelIndex(idx, shape_cuda, stride_cuda, ndim);
        memset(byted_data_ptr + numel_index * element_size, 0, element_size);
    }
}

void ZeroCUDA(
    void* data_ptr, const int* shape_cuda, const std::size_t* stride_cuda,
    int ndim, std::size_t numel, std::size_t element_size
) {
    if (numel == 0) {
        return;
    }
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((numel - 1) / threads + 1);
    ZeroKernel<<<blocks, threads>>>(
        data_ptr, shape_cuda, stride_cuda, ndim, element_size, numel
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void CopyKernel(std::byte* dst, const std::byte* src, const int* shape,
                                  const std::size_t* dst_stride, const std::size_t* src_stride,
                                  int ndim, int numel, std::size_t element_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        const auto dst_offset = GetNumelIndex(idx, shape, dst_stride, ndim) * element_size;
        const auto src_offset = GetNumelIndex(idx, shape, src_stride, ndim) * element_size;
        memcpy(dst + dst_offset, src + src_offset, element_size);
    }
}

void CopyCUDA(void* destination, const void* source, const int* shape,
                     const std::size_t* dst_stride, const std::size_t* src_stride,
                     int ndim, std::size_t numel, std::size_t element_size) {
    if (numel == 0) return;
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((numel - 1) / threads + 1);
    CopyKernel<<<blocks, threads>>>(static_cast<std::byte*>(destination),
        static_cast<const std::byte*>(source), shape, dst_stride, src_stride, ndim, numel, element_size);
    CUDA_CHECK(cudaGetLastError());
}

void CopyHostToCUDA(void* destination, const void* source, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
}

void CopyCUDAToHost(void* destination, const void* source, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
}
}
