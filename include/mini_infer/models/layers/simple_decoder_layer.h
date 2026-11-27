#pragma once

#include <memory>

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
    int heads_;
    std::shared_ptr<RMSNorm> attention_norm_;
    std::shared_ptr<Linear> q_;
    std::shared_ptr<Linear> k_;
    std::shared_ptr<Linear> v_;
    std::shared_ptr<Linear> attention_projection_;
    std::shared_ptr<RMSNorm> mlp_norm_;
    std::shared_ptr<Linear> expand_;
    std::shared_ptr<Linear> contract_;
};

}  // namespace mtensor
