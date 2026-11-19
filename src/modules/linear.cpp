#include <stdexcept>
#include <utility>

#include "mini_infer/modules/linear.h"

namespace mtensor {

Linear::Linear(Tensor weight, std::optional<Tensor> bias)
    : weight_(std::move(weight)), bias_(std::move(bias)) {
    if (weight_.dim() != 2) {
        throw std::invalid_argument("Linear: weight must have shape [out_features, in_features]");
    }
    if (bias_ && (bias_->dim() != 1 || bias_->size(0) != weight_.size(0))) {
        throw std::invalid_argument("Linear: bias must have shape [out_features]");
    }
}

Tensor Linear::forward(const Tensor& x) {
    Tensor y = x.matmul(weight_.transpose(0, 1));
    if (bias_) return y + *bias_;
    return y;
}

}  // namespace mtensor
