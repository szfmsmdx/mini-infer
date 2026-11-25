#pragma once

#include <stdexcept>
#include <string>
#ifdef ENABLE_CUDA
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
