#include <memory>
#include <stdexcept>
#include <utility>

#include "mini_infer/ops/activation.h"
#include "mini_infer/autograd.h"
#include "autograd/backward_nodes/elementwise.h"
#include "ops/backend.h"
#include "tensor/check.h"

namespace mtensor {
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
