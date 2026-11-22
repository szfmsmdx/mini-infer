#pragma once

#include <vector>

#include "mini_infer/autograd.h"

namespace mtensor {

class CloneNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class ToDeviceNode : public AutogradNode {
public:
    Device input_device = Device::kCPU;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class ToDtypeNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

}  // namespace mtensor
