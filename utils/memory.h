#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#ifdef MINI_INFER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

/* 定义CUDA错误检查宏 */
#define CUDA_CHECK(call) \
    do { \
        const cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(error)); \
        } \
    } while (false)

namespace mtensor {

void* AllocateCPU(std::size_t bytes);

// share_ptr 自动管理 storage_ 的析构
// void* FreeCPU(void* storage_);

void* AllocateCUDA(std::size_t bytes);
void FreeCUDA(void* storage_) noexcept;

}  // namespace mtensor
