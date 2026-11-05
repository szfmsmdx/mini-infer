#include <stdexcept>

#include "ops/rope.h"

namespace mtensor {

// TODO：检查输入和位置，分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor rope(const Tensor&, const Tensor&, float, int, RopeStyle) {
    throw std::logic_error("rope: not implemented");
}

}  // namespace mtensor