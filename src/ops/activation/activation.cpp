#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/ops/activation.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "tensor/check.h"

namespace mtensor {
namespace {

class SigmoidNode : public AutogradNode {
public:
    SavedTensor saved_input;
    SavedTensor saved_output;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SiluNode : public AutogradNode {
public:
    SavedTensor saved_input;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

std::vector<Tensor> SigmoidNode::backward(const Tensor& output_grad) {
    const Tensor x = saved_input.unpack("SigmoidNode::backward");
    (void)saved_output.unpack("SigmoidNode::backward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    GetKernelTable().sigmoid_backward.call(
        x.device(), "sigmoid backward", x, output_grad, dx);
    return {dx};
}

std::vector<Tensor> SiluNode::backward(const Tensor& output_grad) {
    const Tensor x = saved_input.unpack("SiluNode::backward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    GetKernelTable().silu_backward.call(
        x.device(), "silu backward", x, output_grad, dx);
    return {dx};
}

}  // namespace

Tensor sigmoid(const Tensor& x) {
    // sigmoid(x) = 1 / (1 + e^(-x))
    CheckNotEmpty(x, "sigmoid: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor y(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    GetKernelTable().sigmoid.call(x.device(), "sigmoid", x, y);
    if (floating && ShouldRecordGrad({&x}, y.dtype())) {
        auto node = std::make_shared<SigmoidNode>();
        node->record_inputs({x});
        node->saved_input = SavedTensor(x);
        node->saved_output = SavedTensor(y, /*is_output=*/true);
        y.set_grad_node(std::move(node));
    }
    return y;
}

Tensor silu(const Tensor& x) {
    // silu(x) = x / (1 + e^(-x))
    CheckNotEmpty(x, "silu: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor y(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    GetKernelTable().silu.call(x.device(), "silu", x, y);
    if (floating && ShouldRecordGrad({&x}, y.dtype())) {
        auto node = std::make_shared<SiluNode>();
        node->record_inputs({x});
        node->saved_input = SavedTensor(x);
        y.set_grad_node(std::move(node));
    }
    return y;
}

}  // namespace mtensor
