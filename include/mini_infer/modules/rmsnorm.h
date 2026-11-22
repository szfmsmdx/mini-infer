#pragma once

#include "mini_infer/modules/module.h"

namespace mtensor {

class RMSNorm : public Module {
public:
    RMSNorm(int channels, float eps, Device device = Device::kCPU);
    RMSNorm(Tensor weight, float eps);
    Tensor forward(const Tensor& x) override;

private:
    void init() override;

    int channels_;
    float eps_;
    Device device_;
    Tensor weight_;
};

}  // namespace mtensor
