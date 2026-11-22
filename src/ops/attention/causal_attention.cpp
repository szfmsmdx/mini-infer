#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "mini_infer/ops/attention.h"
#include "mini_infer/autograd.h"
#include "mini_infer/ops/activation.h"
#include "ops/common/attention_shape.h"
#include "ops/common/input_checks.h"
#include "tensor/check.h"

namespace mtensor {

#ifdef ENABLE_CUDA
void CausalAttentionCUDA(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                         Tensor& probabilities, Tensor& output);
void CausalAttentionGradCUDA(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                             const Tensor& probabilities, const Tensor& grad,
                             Tensor& dq, Tensor& dk, Tensor& dv);
#endif

namespace {

void CausalAttentionCPU(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                        Tensor& probabilities, Tensor& output) {
    NogradNode guard;
    const int length = q.size(0);
    const int head_width = q.size(1) / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_width));

    for (int head = 0; head < heads; ++head) {
        const int head_start = head * head_width;
        const Tensor q_head = q.narrow(1, head_start, head_width);
        const Tensor k_head = k.narrow(1, head_start, head_width);
        const Tensor v_head = v.narrow(1, head_start, head_width);
        const Tensor probs_head = probabilities[head];
        const Tensor output_head = output.narrow(1, head_start, head_width);

        for (int row = 0; row < length; ++row) {
            const int visible = row + 1;
            const Tensor q_row = q_head.narrow(0, row, 1);
            const Tensor k_prefix = k_head.narrow(0, 0, visible);
            const Tensor v_prefix = v_head.narrow(0, 0, visible);
            const Tensor scores = q_row.matmul(k_prefix.transpose(0, 1)) * scale;
            const Tensor p = softmax(scores);
            probs_head.narrow(0, row, 1).narrow(1, 0, visible).copy_(p);
            output_head.narrow(0, row, 1).copy_(p.matmul(v_prefix));
        }
    }
}

void CausalAttentionGradCPU(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                            const Tensor& probabilities, const Tensor& grad,
                            Tensor& dq, Tensor& dk, Tensor& dv) {
    NogradNode guard;
    const int length = q.size(0);
    const int head_width = q.size(1) / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_width));

    for (int head = 0; head < heads; ++head) {
        const int head_start = head * head_width;
        const Tensor q_head = q.narrow(1, head_start, head_width);
        const Tensor k_head = k.narrow(1, head_start, head_width);
        const Tensor v_head = v.narrow(1, head_start, head_width);
        const Tensor probs_head = probabilities[head];
        const Tensor grad_head = grad.narrow(1, head_start, head_width);
        const Tensor dq_head = dq.narrow(1, head_start, head_width);
        const Tensor dk_head = dk.narrow(1, head_start, head_width);
        const Tensor dv_head = dv.narrow(1, head_start, head_width);

        for (int row = 0; row < length; ++row) {
            const int visible = row + 1;
            const Tensor q_row = q_head.narrow(0, row, 1);
            const Tensor k_prefix = k_head.narrow(0, 0, visible);
            const Tensor v_prefix = v_head.narrow(0, 0, visible);
            const Tensor grad_row = grad_head.narrow(0, row, 1);
            const Tensor p = probs_head.narrow(0, row, 1).narrow(1, 0, visible);

            // O = P V；softmax 反向后再乘上分数的缩放系数。
            const Tensor dp = grad_row.matmul(v_prefix.transpose(0, 1));
            const Tensor ds = (p * scale) * (dp - (p * dp).sum(-1, /*keepdim=*/true));
            dq_head.narrow(0, row, 1).copy_(ds.matmul(k_prefix));

            // 同一个 K/V 会被多个后续位置使用，按前缀累加梯度。
            Tensor dk_prefix = dk_head.narrow(0, 0, visible);
            Tensor dv_prefix = dv_head.narrow(0, 0, visible);
            dk_prefix.copy_(dk_prefix + ds.transpose(0, 1).matmul(q_row));
            dv_prefix.copy_(dv_prefix + p.transpose(0, 1).matmul(grad_row));
        }
    }
}

class CausalAttentionNode : public AutogradNode {
public:
    int heads = 0;
    Tensor probabilities;
    std::uint64_t versions[3] = {};

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        for (int i = 0; i < 3; ++i) {
            CheckVersion(inputs[i], versions[i], "causal_attention input modified after forward");
        }
        const Tensor& q = inputs[0];
        const Tensor& k = inputs[1];
        const Tensor& v = inputs[2];
        Tensor dq(q.shape(), q.dtype(), q.device());
        Tensor dk(k.shape(), k.dtype(), k.device());
        Tensor dv(v.shape(), v.dtype(), v.device());
        dk.zero_();
        dv.zero_();
#ifdef ENABLE_CUDA
        if (q.device() == Device::kCUDA) {
            const Tensor packed_q = Packed(q), packed_k = Packed(k), packed_v = Packed(v);
            const Tensor packed_grad = Packed(output_grad);
            CausalAttentionGradCUDA(packed_q, packed_k, packed_v, heads, probabilities, packed_grad, dq, dk, dv);
            return {dq, dk, dv};
        }
#endif
        CausalAttentionGradCPU(q, k, v, heads, probabilities, output_grad, dq, dk, dv);
        return {dq, dk, dv};
    }
};

}  // namespace

Tensor causal_attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads) {
    CheckFloatMatrix(q, "causal_attention Q must be a nonempty FP32 matrix");
    CheckNotEmpty(k, "causal_attention K must not be empty");
    CheckNotEmpty(v, "causal_attention V must not be empty");
    CheckSameShape(q, k, "causal_attention K and Q must have the same shape");
    CheckSameShape(q, v, "causal_attention V and Q must have the same shape");
    CheckSameDtype(q, k, "causal_attention K and Q must have the same dtype");
    CheckSameDtype(q, v, "causal_attention V and Q must have the same dtype");
    CheckSameDevice(q, k, "causal_attention K and Q must be on the same device");
    CheckSameDevice(q, v, "causal_attention V and Q must be on the same device");
    HeadWidth(q.size(1), heads);

    Tensor probabilities({heads, q.size(0), q.size(0)}, q.dtype(), q.device());
    Tensor output(q.shape(), q.dtype(), q.device());
#ifdef ENABLE_CUDA
    if (q.device() == Device::kCUDA) {
        const Tensor packed_q = Packed(q), packed_k = Packed(k), packed_v = Packed(v);
        CausalAttentionCUDA(packed_q, packed_k, packed_v, heads, probabilities, output);
    } else
#endif
    {
        CausalAttentionCPU(q, k, v, heads, probabilities, output);
    }
    if (EnableGrad()) {
        auto node = std::make_shared<CausalAttentionNode>();
        node->inputs = {q, k, v};
        node->heads = heads;
        node->probabilities = std::move(probabilities);
        node->versions[0] = q.version();
        node->versions[1] = k.version();
        node->versions[2] = v.version();
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
