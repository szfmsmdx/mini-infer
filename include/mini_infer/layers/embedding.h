#pragma once

#include "mini_infer/modules/module.h"

namespace mtensor {

class Embedding : public Module {
public:
    Embedding(int vocab_size, int channels, Device device = Device::kCPU,
              WeightInitializer initializer = {});

    Tensor forward(const Tensor& indices) override;
    const Tensor& weight() const { return weight_; }

private:
    void init() override;

    int vocab_size_;
    int channels_;
    Device device_;
    WeightInitializer initializer_;
    Tensor weight_;
};

}  // namespace mtensor
