#pragma once

#include "mini_infer/tensor.h"

namespace mtensor {

// FP32 logits [序列, 词表] 与 Int32 标签 [序列] 的平均交叉熵。
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);

}  // namespace mtensor
