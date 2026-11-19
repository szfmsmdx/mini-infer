#include "autograd/backward_nodes/elementwise.h"
#include "mini_infer/ops/elementwise.h"
#include "autograd/backend.h"
#include "tensor/check.h"

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

std::vector<Tensor> SqrtNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_version, "SqrtNode::backward: input modified after forward");
    const Tensor root = sqrt(inputs[0]);
    return {output_grad / (root + root)};
}

std::vector<Tensor> RsqrtNode::backward(const Tensor& output_grad) {
    const Tensor& x = inputs[0];
    CheckVersion(x, input_version, "RsqrtNode::backward: input modified after forward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::kCPU) {
        RsqrtBackwardCPU(x, output_grad, dx);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        RsqrtBackwardCUDA(x, output_grad, dx);
#endif
    } else {
        throw std::logic_error("rsqrt backward: no backend for this device");
    }
    return {dx};
}

std::vector<Tensor> MulNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_versions[0], "MulNode::backward: input modified after forward");
    CheckVersion(inputs[1], input_versions[1], "MulNode::backward: input modified after forward");
    return {mul(output_grad, inputs[1]), mul(output_grad, inputs[0])};
}

std::vector<Tensor> DivNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_versions[0], "DivNode::backward: input modified after forward");
    CheckVersion(inputs[1], input_versions[1], "DivNode::backward: input modified after forward");
    return {div(output_grad, inputs[1]),
            neg(div(mul(output_grad, inputs[0]), mul(inputs[1], inputs[1])))};
}

}  // namespace mtensor
