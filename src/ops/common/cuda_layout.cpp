#include <algorithm>

#include "ops/common/cuda_layout.h"
#include "mini_infer/core/tensor.h"

namespace mtensor {

// Only the active dimensions are copied. No device allocations or borrowed pointers.
CUDALayout PackLayout(const Tensor& tensor) {
    CUDALayout result{};
    result.dim = tensor.dim();
    result.contiguous = tensor.is_contiguous();
    const auto shape = tensor.shape();
    const auto stride = tensor.stride();
    std::copy(shape.begin(), shape.end(), result.shape);
    std::copy(stride.begin(), stride.end(), result.stride);
    return result;
}

}  // namespace mtensor
