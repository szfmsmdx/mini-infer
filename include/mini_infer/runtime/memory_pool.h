#pragma once

#include <cstddef>

namespace mtensor {

struct CUDAMemoryStats {
    std::size_t allocated_bytes = 0;  // 当前由用户 Tensor 持有的字节数
    std::size_t cached_bytes = 0;     // 显存池中空闲待复用的字节数
    std::size_t allocations = 0;      // 累计分配请求次数
    std::size_t cache_hits = 0;       // 命中缓存池免去 cudaMalloc 的次数
};

void EmptyCUDACache();
CUDAMemoryStats GetCUDAMemoryStats();

}  // namespace mtensor
