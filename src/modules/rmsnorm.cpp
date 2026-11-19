#include <stdexcept>
#include <utility>

#include "mini_infer/modules/rmsnorm.h"
#include "mini_infer/ops/norm.h"

namespace mtensor {

RMSNorm::RMSNorm(Tensor weight, float eps)
    : weight_(std::move(weight)), eps_(eps) {
    if (weight_.dim() != 1) {
        throw std::invalid_argument("RMSNorm: weight must be one-dimensional");
    }
}

Tensor RMSNorm::forward(const Tensor& x) {
    return rms_norm(x, weight_, eps_);
}

}  // namespace mtensor
