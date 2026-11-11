#pragma once

#include "autograd/autograd.h"

namespace mtensor {

class AddNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SubNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class NegNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class MulNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class DivNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

}  // namespace mtensor