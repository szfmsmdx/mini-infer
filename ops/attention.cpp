#include <stdexcept>

#include "ops/attention.h"

namespace mtensor {

// TODO：检查 GQA 比例和 shape，分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor causal_attention(const Tensor&, const Tensor&, const Tensor&, float) {
    throw std::logic_error("causal_attention: not implemented");
}

}  // namespace mtensor