#include <cmath>
#include <stdexcept>
#include <utility>

#include "mini_infer/optim/adam.h"
#include "mini_infer/autograd.h"

namespace mtensor {

Adam::Adam(std::vector<Tensor> parameters, float learning_rate,
           float beta1, float beta2, float eps)
    : Optimizer(std::move(parameters), learning_rate),
      beta1_(beta1), beta2_(beta2), eps_(eps), states_(parameters_.size()) {
    if (!std::isfinite(beta1_) || beta1_ < 0.0f || beta1_ >= 1.0f ||
        !std::isfinite(beta2_) || beta2_ < 0.0f || beta2_ >= 1.0f) {
        throw std::invalid_argument("Adam: betas must be finite and in [0, 1)");
    }
    if (!std::isfinite(eps_) || eps_ < 0.0f) {
        throw std::invalid_argument("Adam: eps must be finite and nonnegative");
    }
}

void Adam::step() {
    NogradNode guard;
    ValidateGradients();
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& parameter = parameters_[i];
        if (!parameter.has_grad()) continue;
        State& state = states_[i];
        const Tensor gradient = parameter.grad();
        Tensor first = state.first_moment;
        Tensor second = state.second_moment;
        if (first.empty()) {
            first = full(parameter.shape(), 0, parameter.dtype(), parameter.device());
            second = full(parameter.shape(), 0, parameter.dtype(), parameter.device());
        }

        Tensor next_first = first * beta1_ + gradient * (1.0f - beta1_);
        Tensor next_second = second * beta2_ + (gradient * gradient) * (1.0f - beta2_);
        const std::uint64_t next_step = state.step + 1;
        const double step = static_cast<double>(next_step);
        const float first_correction = static_cast<float>(1.0 - std::pow(beta1_, step));
        const float second_correction = static_cast<float>(1.0 - std::pow(beta2_, step));
        const Tensor denominator = (next_second / second_correction).sqrt() + eps_;
        const Tensor updated =
            parameter - ((next_first / first_correction) / denominator) * learning_rate_;
        parameter.copy_(updated);

        state.first_moment = std::move(next_first);
        state.second_moment = std::move(next_second);
        state.step = next_step;
    }
}

}  // namespace mtensor
