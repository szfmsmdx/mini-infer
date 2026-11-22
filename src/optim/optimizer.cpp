#include <cmath>
#include <stdexcept>
#include <utility>

#include "mini_infer/optim/optimizer.h"
#include "mini_infer/autograd.h"
#include "tensor/check.h"

namespace mtensor {

Optimizer::Optimizer(std::vector<Tensor> parameters, float learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {
    if (!std::isfinite(learning_rate_) || learning_rate_ < 0.0f) {
        throw std::invalid_argument("Optimizer: learning rate must be finite and nonnegative");
    }
    if (parameters_.empty()) {
        throw std::invalid_argument("Optimizer: parameter list must not be empty");
    }
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        const Tensor& parameter = parameters_[i];
        CheckNotEmpty(parameter, "Optimizer: parameter must not be empty");
        if (parameter.dtype() != Dtype::kFloat32) {
            throw std::invalid_argument("Optimizer: only Float32 parameters are supported");
        }
        CheckBroadcastWritable(parameter);
        for (std::size_t j = 0; j < i; ++j) {
            if (parameter.device() == parameters_[j].device() &&
                parameter.data_ptr() == parameters_[j].data_ptr()) {
                throw std::invalid_argument("Optimizer: duplicate parameter data pointer");
            }
        }
    }
}

void Optimizer::zero_grad() {
    NogradNode guard;
    for (auto& parameter : parameters_) {
        parameter.zero_grad();
    }
}

void Optimizer::ValidateGradients() const {
    // 先校验全部梯度，再开始更新参数。
    for (const auto& parameter : parameters_) {
        if (!parameter.has_grad()) continue;
        const Tensor gradient = parameter.grad();
        CheckNotEmpty(gradient, "Optimizer: gradient must not be empty");
        CheckSameShape(parameter, gradient, "Optimizer: gradient shape must match parameter");
        CheckSameDtype(parameter, gradient, "Optimizer: gradient dtype must match parameter");
        CheckSameDevice(parameter, gradient, "Optimizer: gradient device must match parameter");
    }
}

}  // namespace mtensor
