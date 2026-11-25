#pragma once

#include <cstddef>

#include "mini_infer/tensor/layout_view.h"

namespace mtensor {

class Tensor;

struct CUDALayout {
    int dim;
    bool contiguous;
    int shape[kMaxDim];
    std::size_t stride[kMaxDim];
};

CUDALayout PackLayout(const Tensor& tensor);

}  // namespace mtensor
