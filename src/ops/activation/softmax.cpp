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
    SavedTensor saved_probabilities;
    int dim = -1;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        const Tensor probabilities = saved_probabilities.unpack("softmax probabilities");
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
    if (floating && ShouldRecordGrad({&x}, output.dtype())) {
        auto node = std::make_shared<SoftmaxNode>();
        node->record_inputs({x});
        node->saved_probabilities = SavedTensor(probabilities, /*is_output=*/true);
        node->dim = dim;
        output.set_grad_node(node);
    }
    return output;
}

}  // namespace mtensor
