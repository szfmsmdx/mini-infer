#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/ops/position_embedding.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "ops/common/attention_shape.h"
#include "ops/common/input_checks.h"

namespace mtensor {

namespace {

Tensor ApplyRoPE(const Tensor& x, int head_width, float base,
                 bool inverse, int pos_offset) {
    NogradNode guard;
    Tensor output(x.shape(), x.dtype(), x.device());
    const Tensor packed = Packed(x);
    GetKernelTable().rope.call(
        x.device(), "rope", packed, head_width, base, inverse, pos_offset, output);
    return output;
}

class RoPENode : public AutogradNode {
public:
    int head_width = 0;
    float base = 10000.0f;
    int pos_offset = 0;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        return {ApplyRoPE(output_grad, head_width, base, true, pos_offset)};
    }
};

}  // namespace

Tensor rope(const Tensor& x, int heads, float base, int pos_offset) {
    CheckFloatMatrix(x, "rope input must be a nonempty FP32 matrix");
    const int head_width = HeadWidth(x.size(1), heads);
    if (head_width % 2 != 0 || !std::isfinite(base) || base <= 0.0f) {
        throw std::invalid_argument("rope requires an even head width and a finite positive base");
    }
    if (pos_offset < 0) {
        throw std::invalid_argument("rope: pos_offset must be nonnegative");
    }
    Tensor output = ApplyRoPE(x, head_width, base, false, pos_offset);
    if (ShouldRecordGrad({&x}, output.dtype())) {
        auto node = std::make_shared<RoPENode>();
        node->record_inputs({x});
        node->head_width = head_width;
        node->base = base;
        node->pos_offset = pos_offset;
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
