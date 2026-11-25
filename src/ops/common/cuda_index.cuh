#pragma once

#include <cstddef>

#include "ops/common/cuda_layout.h"

namespace mtensor {

// 把逻辑下标转换成输入视图里的元素偏移。
__device__ inline std::size_t GetNumelIndex(
    std::size_t index, const CUDALayout& layout
) {
    if (layout.contiguous) return index;
    std::size_t offset = 0;
    for (int dim = layout.dim - 1; dim >= 0; --dim) {
        offset += (index % layout.shape[dim]) * layout.stride[dim];
        index /= layout.shape[dim];
    }
    return offset;
}

}  // namespace mtensor
