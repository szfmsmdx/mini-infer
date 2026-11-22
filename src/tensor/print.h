#pragma once

#include <cstddef>
#include <vector>

#include "mini_infer/dtype.h"

namespace mtensor {

void PrintCPU(const void* data_ptr, const std::vector<int>& shape,
              const std::vector<std::size_t>& stride, Dtype dtype, bool is_empty);

}  // namespace mtensor
