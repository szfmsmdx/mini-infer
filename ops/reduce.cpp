#include <stdexcept>
#include <utility>

#include "ops/reduce.h"

namespace mtensor {

// TODO：检查输入、分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor sum(const Tensor&, int, bool) {
    throw std::logic_error("sum: not implemented");
}

Tensor mean(const Tensor&, int, bool) {
    throw std::logic_error("mean: not implemented");
}

Tensor max(const Tensor&, int, bool) {
    throw std::logic_error("max: not implemented");
}

Tensor softmax(const Tensor&, int) {
    throw std::logic_error("softmax: not implemented");
}

Tensor argmax(const Tensor&, int, bool) {
    throw std::logic_error("argmax: not implemented");
}

std::pair<Tensor, Tensor> topk(const Tensor&, int, int) {
    throw std::logic_error("topk: not implemented");
}

}  // namespace mtensor