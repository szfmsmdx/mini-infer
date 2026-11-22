#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/ops/position_embedding.h"
#include "mini_infer/autograd.h"
#include "ops/common/attention_shape.h"
#include "ops/common/input_checks.h"

namespace mtensor {

#ifdef ENABLE_CUDA
void RoPECUDA(const Tensor& x, int head_width, float base, bool inverse, Tensor& output);
#endif

namespace {

void RoPECPU(const Tensor& x, int head_width, float base, bool inverse, Tensor& output) {
    const int length = x.size(0);
    const int heads = x.size(1) / head_width;
    const int pairs_per_head = head_width / 2;
    const std::size_t coefficient_count = static_cast<std::size_t>(length) * pairs_per_head;
    std::vector<float> cosines(coefficient_count), sines(coefficient_count);
    for (int row = 0; row < length; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * pairs_per_head;
        for (int pair = 0; pair < pairs_per_head; ++pair) {
            const float exponent = -static_cast<float>(2 * pair) / head_width;
            const float angle = row * std::pow(base, exponent);
            cosines[offset + pair] = std::cos(angle);
            sines[offset + pair] = inverse ? -std::sin(angle) : std::sin(angle);
        }
    }

    // 每个头共用同一组系数，沿 head 维广播。
    const std::vector<int> coefficient_shape{length, 1, pairs_per_head, 1};
    const Tensor cosine(cosines.data(), coefficient_shape, x.dtype(), x.device());
    const Tensor sine(sines.data(), coefficient_shape, x.dtype(), x.device());
    const Tensor pairs = x.reshape({length, heads, pairs_per_head, 2});
    const Tensor first = pairs.narrow(3, 0, 1);
    const Tensor second = pairs.narrow(3, 1, 1);
    Tensor rotated = output.reshape(pairs.shape());
    rotated.narrow(3, 0, 1).copy_(first * cosine - second * sine);
    rotated.narrow(3, 1, 1).copy_(first * sine + second * cosine);
}

Tensor ApplyRoPE(const Tensor& x, int head_width, float base, bool inverse) {
    NogradNode guard;
    Tensor output(x.shape(), x.dtype(), x.device());
#ifdef ENABLE_CUDA
    if (x.device() == Device::kCUDA) {
        const Tensor packed = Packed(x);
        RoPECUDA(packed, head_width, base, inverse, output);
        return output;
    }
#endif
    RoPECPU(x, head_width, base, inverse, output);
    return output;
}

class RoPENode : public AutogradNode {
public:
    int head_width = 0;
    float base = 10000.0f;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        return {ApplyRoPE(output_grad, head_width, base, true)};
    }
};

}  // namespace

Tensor rope(const Tensor& x, int heads, float base) {
    CheckFloatMatrix(x, "rope input must be a nonempty FP32 matrix");
    const int head_width = HeadWidth(x.size(1), heads);
    if (head_width % 2 != 0 || !std::isfinite(base) || base <= 0.0f) {
        throw std::invalid_argument("rope requires an even head width and a finite positive base");
    }
    Tensor output = ApplyRoPE(x, head_width, base, false);
    if (EnableGrad()) {
        auto node = std::make_shared<RoPENode>();
        node->inputs = {x};
        node->head_width = head_width;
        node->base = base;
        output.set_grad_node(std::move(node));
    }
    return output;
}

}  // namespace mtensor
