#include <stdexcept>

#include "autograd/backward_nodes/tensor_ops.h"

namespace mtensor {

std::vector<Tensor> CloneNode::backward(const Tensor& output_grad) {
    return { output_grad };
}

std::vector<Tensor> ToDeviceNode::backward(const Tensor& output_grad) {
    // auto_grad 引擎确保不会循环建图
    return {output_grad.to(input_device)};
}

std::vector<Tensor> TransposeNode::backward(const Tensor& output_grad) {
    return { output_grad.transpose(dim0, dim1) };
}

std::vector<Tensor> PermuteNode::backward(const Tensor& output_grad) {
    std::vector<int> origin_dims = dims;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        origin_dims[dims[i]] = i;
    }

    return { output_grad.permute(origin_dims) };
}

std::vector<Tensor> BroadcastNode::backward(const Tensor& output_grad) {
    Tensor res = output_grad;
    const auto& input_shape = inputs[0].shape();
    const int leading = output_grad.dim() - inputs[0].dim();
    for (int i = output_grad.dim() - 1; i >= 0; --i) {
        if(i >= leading && input_shape[i - leading] == 1){
            res = res.sum(i, true);
        }
        if(i < leading)
            res = res.sum(i, false);
    }
    return { res };
}

}  // namespace mtensor
