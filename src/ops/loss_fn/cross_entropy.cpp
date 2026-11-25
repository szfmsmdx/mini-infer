#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/ops/loss_fn.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "mini_infer/ops/activation.h"
#include "ops/common/input_checks.h"
#include "tensor/check.h"

namespace mtensor {

namespace {

class CrossEntropyNode : public AutogradNode {
public:
    SavedTensor saved_logits;
    SavedTensor saved_targets;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        const Tensor logits = saved_logits.unpack("cross_entropy logits");
        const Tensor targets = saved_targets.unpack("cross_entropy targets");
        const Tensor ids = Packed(targets);
        const Tensor packed_logits = Packed(logits);
        Tensor logits_grad(logits.shape(), logits.dtype(), logits.device());
        GetKernelTable().cross_entropy_backward.call(
            logits.device(), "cross_entropy backward",
            packed_logits, ids, output_grad, logits_grad);
        return {logits_grad};
    }
};

}  // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    CheckFloatMatrix(logits, "cross_entropy logits must be a nonempty FP32 matrix");
    const Tensor ids = CheckAndPackIndices(targets, logits, logits.size(1));
    if (ids.size(0) != logits.size(0)) {
        throw std::invalid_argument("cross_entropy requires one target per logits row");
    }
    Tensor output;
    {
        NogradNode guard;
        const Tensor packed_logits = Packed(logits);
        Tensor row_losses({logits.size(0)}, logits.dtype(), logits.device());
        GetKernelTable().cross_entropy_rows.call(
            logits.device(), "cross_entropy", packed_logits, ids, row_losses);
        output = row_losses.mean();
    }
    if (ShouldRecordGrad({&logits}, output.dtype())) {
        auto node = std::make_shared<CrossEntropyNode>();
        node->record_inputs({logits});
        node->saved_logits = SavedTensor(logits);
        node->saved_targets = SavedTensor(targets);
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
