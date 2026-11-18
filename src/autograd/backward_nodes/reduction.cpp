#include <stdexcept>

#include "autograd/backward_nodes/reduction.h"

namespace mtensor {

std::vector<Tensor> SumNode::backward(const Tensor& output_grad) {
    const auto& input_shape = inputs[0].shape();

    if (dim == -1) {
        return {output_grad.broadcast(input_shape)};
    }

    auto grad_shape = input_shape;
    grad_shape[dim] = 1;

    Tensor grad = output_grad.reshape(grad_shape);
    return {grad.broadcast(input_shape)};
}

}  // namespace mtensor
