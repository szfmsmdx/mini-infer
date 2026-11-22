#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/dtype.h"

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

// 分配数据存储，并绑定对应设备的释放函数。
std::shared_ptr<void> AllocateStorage(std::size_t bytes, Device device);

// share_ptr 自动管理 storage_ 的析构
// void* FreeCPU(void* storage_);

void* AllocateCUDA(std::size_t bytes);
void FreeCUDA(void* storage_) noexcept;

}  // namespace mtensor
