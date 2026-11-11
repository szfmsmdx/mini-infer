#include <stdexcept>

#include "autograd/backward_nodes/reduction.h"

namespace mtensor {

std::vector<Tensor> SumNode::backward(const Tensor& output_grad) {
    return {full(
        /*shape=*/inputs[0].shape(),
        /*value=*/output_grad,
        /*dtype=*/inputs[0].dtype(),
        /*device=*/inputs[0].device()
    )};
}

}  // namespace mtensor
