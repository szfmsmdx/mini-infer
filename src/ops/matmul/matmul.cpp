#include <memory>
#include <stdexcept>
#include <utility>
#include <cstdint>

#include "mini_infer/ops/matmul.h"
#include "ops/matmul/kernels.h"
#include "tensor/check.h"
#include "mini_infer/autograd.h"

namespace mtensor {

class MatmulNode : public AutogradNode {
public:
    std::uint64_t input_versions[2] = {};
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};


Tensor matmul(const Tensor& a, const Tensor& b) {
    CheckNotEmpty(a, "matmul: left input must not be empty");
    CheckNotEmpty(b, "matmul: right input must not be empty");
    CheckSameDtype(a, b, "matmul: inputs must have the same dtype");
    CheckSameDevice(a, b, "matmul: inputs must be on the same device");
    if (a.dim() != 2 || b.dim() != 2) {
        throw std::invalid_argument("matmul: inputs must be two-dimensional");
    }
    if (a.size(1) != b.size(0)) {
        throw std::invalid_argument("matmul: inner dimensions must match");
    }

    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);
    Tensor output({m, n}, a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        MatmulCPU(a.data_ptr(), b.data_ptr(), output.data_ptr(), a.dtype(),
                  m, k, n, a.stride(), b.stride());
#ifdef ENABLE_CUDA
    } else if (a.device() == Device::kCUDA) {
        MatmulCUDA(a.data_ptr(), b.data_ptr(), output.data_ptr(), a.dtype(),
                   m, k, n, a.stride(), b.stride());
#endif
    } else {
        throw std::logic_error("matmul: no backend implemented for this device");
    }

    if (EnableGrad() && CheckGradDtype(a.dtype())) {
        auto node = std::make_shared<MatmulNode>();
        node->inputs = {a, b};
        node->input_versions[0] = a.version();
        node->input_versions[1] = b.version();
        output.set_grad_node(std::move(node));
    }
    return output;
}

Tensor Tensor::matmul(const Tensor& other) const {
    return mtensor::matmul(*this, other);
}


std::vector<Tensor> MatmulNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_versions[0], "MatmulNode::backward: left input modified after forward");
    CheckVersion(inputs[1], input_versions[1], "MatmulNode::backward: right input modified after forward");
    return {
        output_grad.matmul(inputs[1].transpose(0, 1)), inputs[0].transpose(0, 1).matmul(output_grad)
    };
}

}  // namespace mtensor
