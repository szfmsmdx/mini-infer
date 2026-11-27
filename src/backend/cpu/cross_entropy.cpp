#include <cstdint>

#include "mini_infer/ops/elementwise.h"
#include "mini_infer/tensor/tensor.h"
#include "ops/activation/kernels.h"

namespace mtensor {

void CrossEntropyCPU(const Tensor& logits, const Tensor& targets, Tensor& row_losses) {
    // 写出每行的损失，取平均由调用层负责。
    const auto* ids = static_cast<const std::int32_t*>(targets.data_ptr());
    Tensor target_logits(targets.shape(), logits.dtype(), logits.device());
    for (int row = 0; row < logits.size(0); ++row) {
        target_logits[row].copy_(logits[row][ids[row]]);
    }

    // 先减每行最大值再取指数，避免大 logit 使 exp 溢出。
    const Tensor row_max = logits.max(-1, /*keepdim=*/true);
    const Tensor log_partition = (logits - row_max).exp().sum(-1).log();
    add_out(row_max.squeeze(1) - target_logits, log_partition, row_losses);
}

void CrossEntropyBackwardCPU(const Tensor& logits, const Tensor& targets, const Tensor& grad, Tensor& dx) {
    Tensor prob(logits.shape(), logits.dtype(), logits.device());
    SoftmaxCPU(logits, 1, prob);

    // softmax - one_hot(target)：每行只改目标类别，不用构造 one-hot 张量。
    const auto* ids = static_cast<const std::int32_t*>(targets.data_ptr());
    for (int row = 0; row < logits.size(0); ++row) {
        Tensor target_prob = prob[row][ids[row]];
        target_prob.copy_(target_prob - 1.0f);
    }

    const Tensor scale = grad / logits.size(0);
    mul_out(prob, scale.broadcast(prob.shape()), dx);
}

}  // namespace mtensor
