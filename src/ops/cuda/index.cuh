#pragma once

#include <cstddef>

namespace mtensor {

// 将逻辑下标转换为存储中的元素偏移；shape 和 stride 位于 GPU。
__device__ inline std::size_t GetNumelIndex(
    std::size_t index, const int* shape, const std::size_t* stride, int ndim
) {
    std::size_t offset = 0;
    for (int dim = ndim - 1; dim >= 0; --dim) {
        offset += (index % shape[dim]) * stride[dim];
        index /= shape[dim];
    }
    return offset;
}

}  // namespace mtensor
