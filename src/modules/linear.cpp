#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/modules/linear.h"
#include "mini_infer/init.h"

namespace mtensor {

Linear::Linear(int in_features, int out_features, bool bias, Device device,
               WeightInitializer initializer)
    : Module("Linear"),
      in_features_(in_features), out_features_(out_features), use_bias_(bias), device_(device),
      initializer_(std::move(initializer)) {
    init();
}

Linear::Linear(Tensor weight, std::optional<Tensor> bias)
    : Module("Linear"), use_bias_(bias.has_value()), device_(weight.device()),
      weight_(std::move(weight)), bias_(std::move(bias)) {
    if (weight_.dim() != 2) {
        throw std::invalid_argument("Linear: weight must have shape [out_features, in_features]");
    }
    in_features_ = weight_.size(1);
    out_features_ = weight_.size(0);
    if (bias_ && (bias_->dim() != 1 || bias_->size(0) != out_features_)) {
        throw std::invalid_argument("Linear: bias must have shape [out_features]");
    }
    register_parameters();
}

void Linear::init() {
    if (in_features_ <= 0 || out_features_ <= 0) {
        throw std::invalid_argument("Linear: in_features and out_features must be positive");
    }

    std::vector<float> values(static_cast<std::size_t>(out_features_) * in_features_);
    if (initializer_) {
        initializer_(values.data(), values.size());
        initializer_ = {};
    } else {
        thread_local std::mt19937 rng(std::random_device{}());
        // a = sqrt(5) 时，均匀分布的边界为 1 / sqrt(fan_in)。
        weight_init::kaiming_uniform_(values.data(), rng, in_features_, out_features_, std::sqrt(5.0));
    }
    weight_ = Tensor(values.data(), {out_features_, in_features_}, Dtype::kFloat32, device_);
    weight_.requires_grad_(true);
    if (use_bias_) {
        bias_.emplace(full({out_features_}, 0.0f, Dtype::kFloat32, device_));
        bias_->requires_grad_(true);
    }
    register_parameters();
}

void Linear::register_parameters() {
    register_parameter("weight", weight_);
    if (bias_) register_parameter("bias", *bias_);
}

Tensor Linear::forward(const Tensor& x) {
    Tensor y = x.matmul(weight_.transpose(0, 1));
    if (bias_) return y + *bias_;
    return y;
}

}  // namespace mtensor
