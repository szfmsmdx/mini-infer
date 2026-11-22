#pragma once

#include <optional>

#include "mini_infer/modules/module.h"

namespace mtensor {

class Linear : public Module {
public:
    // 按尺寸创建可训练的 Float32 参数。
    Linear(int in_features, int out_features, bool bias = true, Device device = Device::kCPU,
           WeightInitializer initializer = {});

    // weight: [out_features, in_features]；可选 bias: [out_features]。
    explicit Linear(Tensor weight, std::optional<Tensor> bias = std::nullopt);

    // x: [batch, in_features] -> [batch, out_features]
    Tensor forward(const Tensor& x) override;

private:
    void init() override;
    void register_parameters();

    int in_features_;
    int out_features_;
    bool use_bias_;
    Device device_;
    WeightInitializer initializer_;
    Tensor weight_;
    std::optional<Tensor> bias_;
};

}  // namespace mtensor
