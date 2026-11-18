#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ops/backend.h"
#include "ops/cuda/index.cuh"
#include "utils/memory.h"

namespace mtensor {

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

__global__ void FillFromDeviceKernel(
    std::byte* destination, const std::byte* value, std::size_t numel,
    std::size_t element_size, const int* shape, const std::size_t* stride, int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        const std::size_t offset = GetNumelIndex(idx, shape, stride, ndim) * element_size;
        for (std::size_t i = 0; i < element_size; ++i) {
            destination[offset + i] = value[i];
        }
    }
}

void FillCUDAFromDevice(void* device_ptr, std::size_t numel, std::size_t element_size,
                        const void* device_value, const int* shape,
                        const std::size_t* stride, int ndim) {
    if (numel == 0) return;
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((numel - 1) / threads + 1);
    FillFromDeviceKernel<<<blocks, threads>>>(static_cast<std::byte*>(device_ptr),
        static_cast<const std::byte*>(device_value), numel, element_size, shape, stride, ndim);
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

}  // namespace mtensor
