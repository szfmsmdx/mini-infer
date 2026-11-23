#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/ops/activation.h"
#include "mini_infer/autograd.h"
#include "ops/activation/kernels.h"
#include "tensor/check.h"

namespace mtensor {
namespace {

class SigmoidNode : public AutogradNode {
public:
    Tensor y;
    std::uint64_t input_version = 0;
    std::uint64_t output_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SiluNode : public AutogradNode {
public:
    std::uint64_t input_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

std::vector<Tensor> SigmoidNode::backward(const Tensor& output_grad) {
    const Tensor& x = inputs[0];
    CheckVersion(x, input_version, "SigmoidNode::backward: input modified after forward");
    CheckVersion(y, output_version, "SigmoidNode::backward: output modified after forward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::kCPU) {
        SigmoidBackwardCPU(x, output_grad, dx);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        SigmoidBackwardCUDA(x, output_grad, dx);
#endif
    } else {
        throw std::logic_error("sigmoid backward: no backend for this device");
    }
    return {dx};
}

std::vector<Tensor> SiluNode::backward(const Tensor& output_grad) {
    const Tensor& x = inputs[0];
    CheckVersion(x, input_version, "SiluNode::backward: input modified after forward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::kCPU) {
        SiluBackwardCPU(x, output_grad, dx);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        SiluBackwardCUDA(x, output_grad, dx);
#endif
    } else {
        throw std::logic_error("silu backward: no backend for this device");
    }
    return {dx};
}

}  // namespace

Tensor sigmoid(const Tensor& x) {
    // sigmoid(x) = 1 / (1 + e^(-x))
    CheckNotEmpty(x, "sigmoid: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor y(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    if (x.device() == Device::kCPU) {
        SigmoidCPU(x, y);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        SigmoidCUDA(x, y);
#endif
    } else {
        throw std::logic_error("sigmoid: no backend for this device");
    }
    if (EnableGrad() && floating) {
        auto node = std::make_shared<SigmoidNode>();
        node->inputs = {x};
        node->input_version = x.version();
        node->y = y.detach();
        node->output_version = y.version();
        y.set_grad_node(std::move(node));
    }
    return y;
}

Tensor silu(const Tensor& x) {
    // silu(x) = x / (1 + e^(-x))
    CheckNotEmpty(x, "silu: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor y(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    if (x.device() == Device::kCPU) {
        SiluCPU(x, y);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        SiluCUDA(x, y);
#endif
    } else {
        throw std::logic_error("silu: no backend for this device");
    }
    if (EnableGrad() && floating) {
        auto node = std::make_shared<SiluNode>();
        node->inputs = {x};
        node->input_version = x.version();
        y.set_grad_node(std::move(node));
    }
    return y;
}

}  // namespace mtensor
