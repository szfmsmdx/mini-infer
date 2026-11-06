#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "tensor/tensor.h"
#include "utils/memory.h"

namespace mtensor{

template <typename Element>
__global__ void FillKernel(Element* device_ptr, std::size_t count, Element value){
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < count){
        device_ptr[idx] = value;
    }
}

template <typename Element>
void LaunchTypedFill(
    void* device_ptr,
    std::size_t count,
    const void* host_value,
    unsigned int blocks,
    unsigned int threads
){
    const Element value = *static_cast<const Element*>(host_value); // 不修改原 host_value
    FillKernel<Element><<<blocks, threads>>>(
        static_cast<Element*>(device_ptr), count, value);
}

void LaunchFillKernel(
    void* device_ptr,
    std::size_t count,
    Dtype dtype,
    const void* host_value
){
    if(count == 0){
        return;
    }

    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count - 1) / threads + 1);
    switch (dtype) {
        case Dtype::kFloat16:
        case Dtype::kBFloat16: LaunchTypedFill<std::uint16_t>(device_ptr, count, host_value, blocks, threads); break;
        case Dtype::kFloat32: LaunchTypedFill<float>(device_ptr, count, host_value, blocks, threads); break;
        case Dtype::kInt8: LaunchTypedFill<std::int8_t>(device_ptr, count, host_value, blocks, threads); break;
        case Dtype::kUInt8: LaunchTypedFill<std::uint8_t>(device_ptr, count, host_value, blocks, threads); break;
        case Dtype::kInt16: LaunchTypedFill<std::int16_t>(device_ptr, count, host_value, blocks, threads); break;
        case Dtype::kInt32: LaunchTypedFill<std::int32_t>(device_ptr, count, host_value, blocks, threads); break;
        case Dtype::kInt64: LaunchTypedFill<std::int64_t>(device_ptr, count, host_value, blocks, threads); break;
        default: throw std::invalid_argument("unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}

void CopyHostToCUDA(void* destination, const void* source, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
}

void CopyCUDA(void* destination, const void* source, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice));
}

void CopyCUDAToHost(void* destination, const void* source, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
}

void ZeroCUDA(void* data, std::size_t bytes) {
    CUDA_CHECK(cudaMemset(data, 0, bytes));
}
}
