#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ops/copy/kernels.h"
#include "ops/common/cuda_index.cuh"
#include "device/memory.h"

namespace mtensor {

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

}  // namespace mtensor
