#include <stdexcept>

#include "ops/conv1d.h"

namespace mtensor {

// TODO：检查输入、权重和 bias，分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor depthwise_causal_conv1d(const Tensor&, const Tensor&, const Tensor*) {
    throw std::logic_error("depthwise_causal_conv1d: not implemented");
}

}  // namespace mtensor