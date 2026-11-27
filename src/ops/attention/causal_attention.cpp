#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "mini_infer/ops/attention.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "mini_infer/ops/activation.h"
#include "ops/common/attention_shape.h"
#include "ops/common/input_checks.h"
#include "tensor/check.h"

namespace mtensor {

namespace {

class CausalAttentionNode : public AutogradNode {
public:
    int heads = 0;
    SavedTensor saved_inputs[3];
    SavedTensor saved_softmax_prob;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        const Tensor q = saved_inputs[0].unpack("causal_attention Q");
        const Tensor k = saved_inputs[1].unpack("causal_attention K");
        const Tensor v = saved_inputs[2].unpack("causal_attention V");
        const Tensor prob = saved_softmax_prob.unpack("causal_attention softmax prob");
        Tensor dq(q.shape(), q.dtype(), q.device());
        Tensor dk(k.shape(), k.dtype(), k.device());
        Tensor dv(v.shape(), v.dtype(), v.device());
        dk.zero_();
        dv.zero_();
        const Tensor packed_q = Packed(q), packed_k = Packed(k), packed_v = Packed(v);
        const Tensor packed_grad = Packed(output_grad);
        GetKernelTable().causal_attention_backward.call(
            q.device(), "causal_attention backward", packed_q, packed_k, packed_v,
            heads, prob, packed_grad, dq, dk, dv);
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

    Tensor prob({heads, q.size(0), q.size(0)}, q.dtype(), q.device());
    Tensor output(q.shape(), q.dtype(), q.device());
    const Tensor packed_q = Packed(q), packed_k = Packed(k), packed_v = Packed(v);
    GetKernelTable().causal_attention.call(
        q.device(), "causal_attention", packed_q, packed_k, packed_v,
        heads, prob, output);
    if (ShouldRecordGrad({&q, &k, &v}, output.dtype())) {
        auto node = std::make_shared<CausalAttentionNode>();
        node->record_inputs({q, k, v});
        node->heads = heads;
        node->saved_inputs[0] = SavedTensor(q);
        node->saved_inputs[1] = SavedTensor(k);
        node->saved_inputs[2] = SavedTensor(v);
        node->saved_softmax_prob = SavedTensor(prob, /*is_output=*/true);
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
