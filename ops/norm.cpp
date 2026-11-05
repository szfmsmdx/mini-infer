#include <stdexcept>

#include "ops/norm.h"

namespace mtensor {

// TODO：检查输入、分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor rms_norm(const Tensor&, const Tensor&, float) {
    throw std::logic_error("rms_norm: not implemented");
}

Tensor rms_norm_zero_centered(const Tensor&, const Tensor&, float) {
    throw std::logic_error("rms_norm_zero_centered: not implemented");
}

Tensor rms_norm_gated(const Tensor&, const Tensor&, const Tensor&, float) {
    throw std::logic_error("rms_norm_gated: not implemented");
}

Tensor layer_norm(const Tensor&, const Tensor&, const Tensor&, float) {
    throw std::logic_error("layer_norm: not implemented");
}

}  // namespace mtensor