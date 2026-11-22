#pragma once

#include "mini_infer/tensor.h"

namespace mtensor {

// 对 [序列, 通道] 的每个头按相邻通道配对旋转，位置从 0 开始。
Tensor rope(const Tensor& x, int heads, float base = 10000.0f);

}  // namespace mtensor
