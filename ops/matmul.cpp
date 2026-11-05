#include <stdexcept>

#include "ops/matmul.h"

namespace mtensor {

// TODO：检查输入、分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor matmul(const Tensor&, const Tensor&) {
    throw std::logic_error("matmul: not implemented");
}

Tensor linear(const Tensor&, const Tensor&, const Tensor*) {
    throw std::logic_error("linear: not implemented");
}

}  // namespace mtensor