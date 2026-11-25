#pragma once

#include <cstddef>

#include "mini_infer/tensor/layout_view.h"
#include "mini_infer/tensor/dtype.h"

namespace mtensor {

void PrintCPU(const void* data_ptr, ShapeView shape,
              StrideView stride, Dtype dtype, bool is_empty);

}  // namespace mtensor
