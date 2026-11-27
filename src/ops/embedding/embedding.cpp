#include <memory>
#include <utility>
#include <vector>

#include "mini_infer/ops/embedding.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "ops/common/input_checks.h"
#include "tensor/check.h"

namespace mtensor {

namespace {

class EmbeddingNode : public AutogradNode {
public:
    SavedTensor saved_indices;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        const Tensor indices = saved_indices.unpack("embedding indices");
        const Tensor& weight = inputs[0];
        const Tensor ids = Packed(indices);
        Tensor weight_grad(weight.shape(), weight.dtype(), weight.device());
        weight_grad.zero_();
        const Tensor grad = Packed(output_grad);
        GetKernelTable().embedding_backward.call(
            weight.device(), "embedding backward", grad, ids, weight_grad);
        return {weight_grad};
    }
};

}  // namespace

Tensor embedding(const Tensor& weight, const Tensor& indices) {
    CheckFloatMatrix(weight, "embedding weight must be a nonempty FP32 matrix");
    const Tensor ids = CheckAndPackIndices(indices, weight, weight.size(0));
    Tensor output({ids.size(0), weight.size(1)}, weight.dtype(), weight.device());
    const Tensor packed_weight = Packed(weight);
    GetKernelTable().embedding.call(
        weight.device(), "embedding", packed_weight, ids, output);
    if (ShouldRecordGrad({&weight}, output.dtype())) {
        auto node = std::make_shared<EmbeddingNode>();
        node->record_inputs({weight});
        node->saved_indices = SavedTensor(indices);
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
