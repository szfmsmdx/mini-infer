#pragma once

#include <vector>

#include "mini_infer/modules/module.h"
#include "mini_infer/modules/linear.h"
#include "mini_infer/modules/rmsnorm.h"

namespace mtensor {

class SimpleDecoderLayer : public Module {
public:
    SimpleDecoderLayer(int channels, int heads, float norm_eps, Device device = Device::kCPU,
                       const WeightInitializer& initializer = {},
                       const WeightInitializer& residual_initializer = {});

    Tensor forward(const Tensor& x) override;

private:
    void append_child_parameters(std::vector<Tensor>& result) const override;

    int heads_;
    RMSNorm attention_norm_;
    Linear q_;
    Linear k_;
    Linear v_;
    Linear attention_projection_;
    RMSNorm mlp_norm_;
    Linear expand_;
    Linear contract_;
};

}  // namespace mtensor
