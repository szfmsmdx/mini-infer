#include <stdexcept>

#include "ops/activation.h"

namespace mtensor {

// TODO：检查输入、分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor silu(const Tensor&) {
    throw std::logic_error("silu: not implemented");
}

Tensor sigmoid(const Tensor&) {
    throw std::logic_error("sigmoid: not implemented");
}

Tensor softplus(const Tensor&) {
    throw std::logic_error("softplus: not implemented");
}

Tensor exp(const Tensor&) {
    throw std::logic_error("exp: not implemented");
}

}  // namespace mtensor