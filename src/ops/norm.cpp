#include <stdexcept>

#include "mini_infer/ops/norm.h"

namespace mtensor {

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.dim() == 0) {
        throw std::invalid_argument("rms_norm: x must have a feature dimension");
    }
    if (weight.dim() != 1 || weight.size(0) != x.size(x.dim() - 1)) {
        throw std::invalid_argument("rms_norm: weight must be a vector matching the last dimension of x");
    }
    // y = weight * x / sqrt(mean(x²) + eps)
    auto inv_rms = ((x * x).mean(-1, true) + eps).rsqrt();
    return (x * inv_rms) * weight;
}

}  // namespace mtensor
