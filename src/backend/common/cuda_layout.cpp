#include <algorithm>

#include "backend/common/cuda_layout.h"
#include "mini_infer/tensor/tensor.h"

namespace mtensor {

// 只拷贝有效的维度。不做设备分配，也不借用指针。
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
