#include "autograd/backward_nodes/matmul.h"
#include "tensor/check.h"

namespace mtensor {

std::vector<Tensor> MatmulNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_versions[0], "MatmulNode::backward: left input modified after forward");
    CheckVersion(inputs[1], input_versions[1], "MatmulNode::backward: right input modified after forward");
    return {
        output_grad.matmul(inputs[1].transpose(0, 1)), inputs[0].transpose(0, 1).matmul(output_grad)
    };
}

}  // namespace mtensor
