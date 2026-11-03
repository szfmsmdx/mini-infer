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
    if(dtype == Dtype::kBFloat16 || dtype == Dtype::kFloat16){
        LaunchTypedFill<std::uint16_t>(
            device_ptr, count, host_value, blocks, threads);
    }else{
        MTENSOR_DISPATCH_DTYPE(dtype, Element, {
            LaunchTypedFill<Element>(device_ptr, count, host_value, blocks, threads);
        });
    }

    CUDA_CHECK(cudaGetLastError());
}
}
