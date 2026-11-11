#include "autograd/backward_nodes/elementwise.h"
#include "ops/elementwise.h"

namespace mtensor {

std::vector<Tensor> AddNode::backward(const Tensor& output_grad) {
    return {output_grad, output_grad};
}

std::vector<Tensor> SubNode::backward(const Tensor& output_grad) {
    return {output_grad, neg(output_grad)};
}

std::vector<Tensor> NegNode::backward(const Tensor& output_grad) {
    return {neg(output_grad)};
}

std::vector<Tensor> MulNode::backward(const Tensor& output_grad) {
    return {mul(output_grad, inputs[1]), mul(output_grad, inputs[0])};
}

std::vector<Tensor> DivNode::backward(const Tensor& output_grad) {
    return {div(output_grad, inputs[1]),
            neg(div(mul(output_grad, inputs[0]), mul(inputs[1], inputs[1])))};
}

}  // namespace mtensor