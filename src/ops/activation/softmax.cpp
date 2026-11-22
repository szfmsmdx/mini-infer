#include <cstdint>
#include <memory>
#include <vector>

#include "mini_infer/ops/activation.h"
#include "mini_infer/autograd.h"
#include "tensor/check.h"

namespace mtensor {

namespace {

class SoftmaxNode : public AutogradNode {
public:
    Tensor probabilities;
    std::uint64_t probabilities_version = 0;
    int dim = -1;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        CheckVersion(probabilities, probabilities_version, "softmax: probabilities modified after forward");
        const Tensor grad = output_grad.to(Dtype::kFloat32);
        const Tensor weighted = probabilities * grad;
        const Tensor input_grad = weighted - probabilities * weighted.sum(dim, /*keepdim=*/true);
        return {input_grad.to(inputs[0].dtype())};
    }
};

}  // namespace

Tensor softmax(const Tensor& x, int dim) {
    CheckNotEmpty(x, "softmax: input must not be empty");
    if (dim < 0) dim += x.dim();
    CheckDim(dim, x.shape().size(), "softmax: dim out of range");

    const bool floating = CheckGradDtype(x.dtype());
    Tensor probabilities;
    Tensor output;
    {
        NogradNode guard;
        const Tensor input = x.to(Dtype::kFloat32);
        const Tensor weights = (input - input.max(dim, /*keepdim=*/true)).exp();
        probabilities = weights / weights.sum(dim, /*keepdim=*/true);
        output = floating ? probabilities.to(x.dtype()) : probabilities;
    }
    if (EnableGrad() && floating) {
        auto node = std::make_shared<SoftmaxNode>();
        node->inputs = {x};
        node->probabilities = probabilities.detach();
        node->probabilities_version = probabilities.version();
        node->dim = dim;
        output.set_grad_node(node);
    }
    return output;
}

}  // namespace mtensor
