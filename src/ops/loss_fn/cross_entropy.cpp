#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/ops/loss_fn.h"
#include "mini_infer/autograd.h"
#include "mini_infer/ops/activation.h"
#include "ops/common/input_checks.h"
#include "tensor/check.h"

namespace mtensor {

#ifdef ENABLE_CUDA
void CrossEntropyCUDA(const Tensor& logits, const Tensor& targets, Tensor& row_losses);
void CrossEntropyGradCUDA(const Tensor& logits, const Tensor& targets,
                          const Tensor& grad, Tensor& logits_grad);
#endif

namespace {

Tensor CrossEntropyCPU(const Tensor& logits, const Tensor& targets) {
    const auto* ids = static_cast<const std::int32_t*>(targets.data_ptr());
    Tensor target_logits(targets.shape(), logits.dtype(), logits.device());
    for (int row = 0; row < logits.size(0); ++row) {
        target_logits[row].copy_(logits[row][ids[row]]);
    }
    const Tensor maximum = logits.max(-1, /*keepdim=*/true);
    const Tensor log_partition = (logits - maximum).exp().sum(-1).log();
    return (maximum.squeeze(1) - target_logits) + log_partition;
}

Tensor CrossEntropyGradCPU(const Tensor& logits, const Tensor& targets, const Tensor& grad) {
    NogradNode guard;
    const int length = logits.size(0);
    const auto* ids = static_cast<const std::int32_t*>(targets.data_ptr());
    Tensor probabilities = softmax(logits, -1);
    for (int row = 0; row < length; ++row) {
        Tensor target_probability = probabilities[row][ids[row]];
        target_probability.copy_(target_probability - 1.0f);
    }
    return probabilities * (grad / length);
}

class CrossEntropyNode : public AutogradNode {
public:
    Tensor targets;
    std::uint64_t logits_version = 0;
    std::uint64_t target_version = 0;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        CheckVersion(inputs[0], logits_version, "cross_entropy logits modified after forward");
        CheckVersion(targets, target_version, "cross_entropy targets modified after forward");
        const Tensor& logits = inputs[0];
        const Tensor ids = Packed(targets);
#ifdef ENABLE_CUDA
        if (logits.device() == Device::kCUDA) {
            const Tensor packed_logits = Packed(logits);
            Tensor logits_grad(logits.shape(), logits.dtype(), logits.device());
            CrossEntropyGradCUDA(packed_logits, ids, output_grad, logits_grad);
            return {logits_grad};
        }
#endif
        return {CrossEntropyGradCPU(logits, ids, output_grad)};
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
#ifdef ENABLE_CUDA
        if (logits.device() == Device::kCUDA) {
            const Tensor packed_logits = Packed(logits);
            Tensor row_losses({logits.size(0)}, logits.dtype(), logits.device());
            CrossEntropyCUDA(packed_logits, ids, row_losses);
            output = row_losses.mean();
        } else
#endif
        {
            output = CrossEntropyCPU(logits, ids).mean();
        }
    }
    if (EnableGrad()) {
        auto node = std::make_shared<CrossEntropyNode>();
        node->inputs = {logits};
        node->targets = targets;
        node->logits_version = logits.version();
        node->target_version = targets.version();
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
