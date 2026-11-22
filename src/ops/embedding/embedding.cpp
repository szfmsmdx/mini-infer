#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "mini_infer/ops/embedding.h"
#include "mini_infer/autograd.h"
#include "ops/common/input_checks.h"
#include "tensor/check.h"

namespace mtensor {

#ifdef ENABLE_CUDA
void EmbeddingCUDA(const Tensor& weight, const Tensor& indices, Tensor& output);
void EmbeddingGradCUDA(const Tensor& grad, const Tensor& indices, Tensor& weight_grad);
#endif

namespace {

void EmbeddingCPU(const Tensor& weight, const Tensor& indices, Tensor& output) {
    NogradNode guard;
    const auto* ids = static_cast<const std::int32_t*>(indices.data_ptr());
    for (int row = 0; row < indices.size(0); ++row) {
        output[row].copy_(weight[ids[row]]);
    }
}

void EmbeddingGradCPU(const Tensor& grad, const Tensor& indices, Tensor& weight_grad) {
    NogradNode guard;
    const auto* ids = static_cast<const std::int32_t*>(indices.data_ptr());
    for (int row = 0; row < indices.size(0); ++row) {
        Tensor target_row = weight_grad[ids[row]];
        target_row.copy_(target_row + grad[row]);
    }
}

class EmbeddingNode : public AutogradNode {
public:
    Tensor indices;
    std::uint64_t index_version = 0;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        CheckVersion(indices, index_version, "embedding indices modified after forward");
        const Tensor& weight = inputs[0];
        const Tensor ids = Packed(indices);
        Tensor weight_grad(weight.shape(), weight.dtype(), weight.device());
        weight_grad.zero_();
#ifdef ENABLE_CUDA
        if (weight.device() == Device::kCUDA) {
            const Tensor grad = Packed(output_grad);
            EmbeddingGradCUDA(grad, ids, weight_grad);
            return {weight_grad};
        }
#endif
        EmbeddingGradCPU(output_grad, ids, weight_grad);
        return {weight_grad};
    }
};

}  // namespace

Tensor embedding(const Tensor& weight, const Tensor& indices) {
    CheckFloatMatrix(weight, "embedding weight must be a nonempty FP32 matrix");
    const Tensor ids = CheckAndPackIndices(indices, weight, weight.size(0));
    Tensor output({ids.size(0), weight.size(1)}, weight.dtype(), weight.device());
#ifdef ENABLE_CUDA
    if (weight.device() == Device::kCUDA) {
        const Tensor packed_weight = Packed(weight);
        EmbeddingCUDA(packed_weight, ids, output);
    } else
#endif
    {
        EmbeddingCPU(weight, ids, output);
    }
    if (EnableGrad()) {
        auto node = std::make_shared<EmbeddingNode>();
        node->inputs = {weight};
        node->indices = indices;
        node->index_version = indices.version();
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
