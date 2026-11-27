#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/ops/matmul.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "tensor/check.h"

namespace mtensor {

class MatmulNode : public AutogradNode {
public:
    SavedTensor saved_inputs[2];
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

Tensor& matmul_out(const Tensor& a, const Tensor& b, Tensor& output) {
    CheckNotEmpty(a, "matmul_out: left input must not be empty");
    CheckNotEmpty(b, "matmul_out: right input must not be empty");
    CheckNotEmpty(output, "matmul_out: output must not be empty");
    CheckSameDtype(a, b, "matmul_out: inputs must have the same dtype");
    CheckSameDevice(a, b, "matmul_out: inputs must be on the same device");
    CheckSameDtype(a, output, "matmul_out: output dtype must match");
    CheckSameDevice(a, output, "matmul_out: output device must match");
    if (a.dim() != 2 || b.dim() != 2 || output.dim() != 2 ||
        a.size(1) != b.size(0) ||
        output.size(0) != a.size(0) || output.size(1) != b.size(1)) {
        throw std::invalid_argument("matmul_out: dimensions do not match");
    }
    if (!output.is_contiguous() || output.shares_storage(a) || output.shares_storage(b)) {
        throw std::invalid_argument("matmul_out: output must be contiguous and independent");
    }
    output.update_version();
    GetKernelTable().matmul.call(a.device(), "matmul_out", a, b, output);
    return output;
}


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
    const int n = b.size(1);
    Tensor output({m, n}, a.dtype(), a.device());
    matmul_out(a, b, output);

    if (ShouldRecordGrad({&a, &b}, output.dtype())) {
        auto node = std::make_shared<MatmulNode>();
        node->record_inputs({a, b});
        node->saved_inputs[0] = SavedTensor(a);
        node->saved_inputs[1] = SavedTensor(b);
        output.set_grad_node(std::move(node));
    }
    return output;
}

Tensor Tensor::matmul(const Tensor& other) const {
    return mtensor::matmul(*this, other);
}


std::vector<Tensor> MatmulNode::backward(const Tensor& output_grad) {
    const Tensor a = saved_inputs[0].unpack("MatmulNode::backward: left input");
    const Tensor b = saved_inputs[1].unpack("MatmulNode::backward: right input");
    Tensor da(a.shape(), a.dtype(), a.device());
    Tensor db(b.shape(), b.dtype(), b.device());
    GetKernelTable().matmul_backward.call(
        a.device(), "matmul backward", a, b, output_grad, da, db);
    return {da, db};
}

}  // namespace mtensor
