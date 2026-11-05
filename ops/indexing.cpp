#include <stdexcept>

#include "ops/indexing.h"

namespace mtensor {

// TODO：检查输入和索引、分配输出，并按 device 分发到 CPU/CUDA 后端。
Tensor embedding(const Tensor&, const Tensor&) {
    throw std::logic_error("embedding: not implemented");
}

Tensor index_select(const Tensor&, int, const Tensor&) {
    throw std::logic_error("index_select: not implemented");
}

Tensor gather(const Tensor&, int, const Tensor&) {
    throw std::logic_error("gather: not implemented");
}

Tensor scatter_add(const Tensor&, int, const Tensor&, const Tensor&) {
    throw std::logic_error("scatter_add: not implemented");
}

}  // namespace mtensor