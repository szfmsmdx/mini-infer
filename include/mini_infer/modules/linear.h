#pragma once

#include <optional>

#include "mini_infer/modules/module.h"

namespace mtensor {

class Linear : public Module {
public:
    // weight: [out_features, in_features]；可选 bias: [out_features]。
    explicit Linear(Tensor weight, std::optional<Tensor> bias = std::nullopt);

    // x: [batch, in_features] -> [batch, out_features]
    Tensor forward(const Tensor& x) override;

private:
    Tensor weight_;
    std::optional<Tensor> bias_;
};

}  // namespace mtensor
