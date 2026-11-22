#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mini_infer/ops/norm.h"
#include "mini_infer/autograd.h"
#include "ops/norm/kernels.h"
#include "tensor/check.h"

namespace mtensor {
namespace {

Tensor InverseRms(const Tensor& x, int features, float eps) {
    // 保留低精度路径原先的除法顺序；mean() 使用乘倒数。
    return ((x * x).sum(-1, /*keepdim=*/true) / features + eps).rsqrt();
}

// 低精度输入只在输出和梯度写回时舍入，中间计算保持 FP32。
class RMSNormNode : public AutogradNode {
public:
    std::uint64_t input_versions[2] = {};
    float eps = 0.0f;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        const Tensor& x = inputs[0];
        const Tensor& weight = inputs[1];
        CheckVersion(x, input_versions[0], "RMSNormNode::backward: input modified after forward");
        CheckVersion(weight, input_versions[1], "RMSNormNode::backward: weight modified after forward");

#ifdef ENABLE_CUDA
        if (x.device() == Device::kCUDA) {
            Tensor dx(x.shape(), x.dtype(), x.device());
            Tensor dw(weight.shape(), weight.dtype(), weight.device());
            RMSNormBackwardCUDA(x, weight, output_grad, eps, dx, dw);
            return {dx, dw};
        }
#endif
        const int features = weight.size(0);
        const Tensor xf = x.to(Dtype::kFloat32);
        const Tensor wf = weight.to(Dtype::kFloat32);
        const Tensor grad = output_grad.to(Dtype::kFloat32);
        const Tensor inv_rms = InverseRms(xf, features, eps);
        const Tensor grad_weighted = grad * wf;
        const Tensor projection = (grad_weighted * xf).sum(-1, /*keepdim=*/true) / features;
        const Tensor correction = inv_rms * inv_rms * inv_rms * projection;
        const Tensor dx = grad_weighted * inv_rms - xf * correction;

        // 把前导维合成行，在写回低精度前按特征累加。
        const Tensor dw = ((grad * xf) * inv_rms).reshape({-1, features}).sum(0);
        return {dx.to(x.dtype()), dw.to(weight.dtype())};
    }
};

}  // namespace

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.dim() == 0) {
        throw std::invalid_argument("rms_norm: x must have a feature dimension");
    }
    const int features = x.size(x.dim() - 1);
    if (weight.dim() != 1 || weight.size(0) != features) {
        throw std::invalid_argument("rms_norm: weight must be a vector matching the last dimension of x");
    }
    if (x.dtype() != Dtype::kFloat16 && x.dtype() != Dtype::kBFloat16) {
        const Tensor inv_rms = ((x * x).mean(-1, /*keepdim=*/true) + eps).rsqrt();
        return (x * inv_rms) * weight;
    }

    CheckSameDtype(x, weight, "rms_norm: x and weight must have the same dtype");
    CheckSameDevice(x, weight, "rms_norm: x and weight must have the same device");
    CheckNotEmpty(x, "rms_norm: x must not be empty");
    CheckNotEmpty(weight, "rms_norm: weight must not be empty");

    Tensor output;
#ifdef ENABLE_CUDA
    if (x.device() == Device::kCUDA) {
        output = Tensor(x.shape(), x.dtype(), x.device());
        RMSNormCUDA(x, weight, eps, output);
    } else
#endif
    {
        NogradNode guard;
        const Tensor xf = x.to(Dtype::kFloat32);
        const Tensor wf = weight.to(Dtype::kFloat32);
        const Tensor inv_rms = InverseRms(xf, features, eps);
        output = ((xf * inv_rms) * wf).to(x.dtype());
    }
    if (EnableGrad()) {
        auto node = std::make_shared<RMSNormNode>();
        node->inputs = {x, weight};
        node->input_versions[0] = x.version();
        node->input_versions[1] = weight.version();
        node->eps = eps;
        output.set_grad_node(node);
    }
    return output;
}

}  // namespace mtensor
