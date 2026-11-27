#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mini_infer/ops/norm.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "tensor/check.h"

namespace mtensor {
namespace {

class RMSNormNode : public AutogradNode {
public:
    SavedTensor saved_inputs[2];
    float eps = 0.0f;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        const Tensor x = saved_inputs[0].unpack("RMSNormNode::backward: input");
        const Tensor weight = saved_inputs[1].unpack("RMSNormNode::backward: weight");
        Tensor dx(x.shape(), x.dtype(), x.device());
        Tensor dw(weight.shape(), weight.dtype(), weight.device());
        GetKernelTable().rms_norm_backward.call(
            x.device(), "rms_norm backward", x, weight, output_grad, eps, dx, dw);
        return {dx, dw};
    }
};

}  // namespace

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    CheckNotEmpty(x, "rms_norm: x must not be empty");
    CheckNotEmpty(weight, "rms_norm: weight must not be empty");
    CheckFloatingDtype(x.dtype(), "rms_norm: input must have floating dtype");
    if (x.dim() == 0) {
        throw std::invalid_argument("rms_norm: x must have a feature dimension");
    }
    const int features = x.size(x.dim() - 1);
    if (weight.dim() != 1 || weight.size(0) != features) {
        throw std::invalid_argument("rms_norm: weight must be a vector matching the last dimension of x");
    }
    CheckSameDtype(x, weight, "rms_norm: x and weight must have the same dtype");
    CheckSameDevice(x, weight, "rms_norm: x and weight must have the same device");
    Tensor output(x.shape(), x.dtype(), x.device());
    GetKernelTable().rms_norm.call(x.device(), "rms_norm", x, weight, eps, output);
    if (ShouldRecordGrad({&x, &weight}, output.dtype())) {
        auto node = std::make_shared<RMSNormNode>();
        node->record_inputs({x, weight});
        node->saved_inputs[0] = SavedTensor(x);
        node->saved_inputs[1] = SavedTensor(weight);
        node->eps = eps;
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
