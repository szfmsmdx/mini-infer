#pragma once

#include "mini_infer/modules/module.h"

namespace mtensor {

class RMSNorm : public Module {
public:
    RMSNorm(Tensor weight, float eps);
    Tensor forward(const Tensor& x) override;

private:
    Tensor weight_;
    float eps_;
};

}  // namespace mtensor
