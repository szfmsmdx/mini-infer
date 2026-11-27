#include <stdexcept>
#include <utility>

#include "mini_infer/modules/rmsnorm.h"
#include "mini_infer/ops/norm.h"

namespace mtensor {

RMSNorm::RMSNorm(int channels, float eps, Device device)
    : Module("RMSNorm"), channels_(channels), eps_(eps), device_(device) {
    init();
}

RMSNorm::RMSNorm(Tensor weight, float eps)
    : Module("RMSNorm"), eps_(eps), device_(weight.device()), weight_(std::move(weight)) {
    if (weight_.dim() != 1) {
        throw std::invalid_argument("RMSNorm: weight must be one-dimensional");
    }
    channels_ = weight_.size(0);
    register_parameter("weight", weight_);
}

void RMSNorm::init() {
    if (channels_ <= 0) {
        throw std::invalid_argument("RMSNorm: channels must be positive");
    }
    weight_ = full({channels_}, 1.0f, Dtype::kFloat32, device_);
    weight_.requires_grad_(true);
    register_parameter("weight", weight_);
}

Tensor RMSNorm::forward(const Tensor& x) {
    return rms_norm(x, weight_, eps_);
}

}  // namespace mtensor
