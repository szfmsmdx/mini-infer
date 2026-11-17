#pragma once

#include <vector>

#include "autograd/autograd.h"

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

class TransposeNode : public AutogradNode {
public:
    int dim0, dim1;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class PermuteNode : public AutogradNode {
public:
    std::vector<int> dims;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class BroadcastNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

}  // namespace mtensor
