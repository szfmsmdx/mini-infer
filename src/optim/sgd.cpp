#include <cmath>
#include <stdexcept>
#include <utility>

#include "mini_infer/optim/sgd.h"
#include "mini_infer/autograd.h"

namespace mtensor {

SGD::SGD(std::vector<Tensor> parameters, float learning_rate, float momentum)
    : Optimizer(std::move(parameters), learning_rate), momentum_(momentum),
      momentum_buffers_(parameters_.size()) {
    if (!std::isfinite(momentum_) || momentum_ < 0.0f) {
        throw std::invalid_argument("SGD: momentum must be finite and nonnegative");
    }
}

void SGD::step() {
    NogradNode guard;
    ValidateGradients();
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& parameter = parameters_[i];
        if (!parameter.has_grad()) continue;
        Tensor direction = parameter.grad();
        if (momentum_ != 0.0f) {
            if (momentum_buffers_[i].empty()) {
                direction = direction.clone();
            } else {
                direction = momentum_buffers_[i] * momentum_ + direction;
            }
        }

        Tensor updated = parameter - direction * learning_rate_;
        parameter.copy_(updated);
        // 参数成功写回后，再接管本次计算出的动量。
        if (momentum_ != 0.0f) {
            momentum_buffers_[i] = std::move(direction);
        }
    }
}

}  // namespace mtensor
