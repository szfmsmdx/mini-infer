#include <stdexcept>

#include "ops/gated_delta.h"

namespace mtensor {

// TODO：检查输入和 state，分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor gated_delta_rule(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    Tensor&) {
    throw std::logic_error("gated_delta_rule: not implemented");
}

}  // namespace mtensor