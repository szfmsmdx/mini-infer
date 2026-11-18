#pragma once

#include <cstdint>

#include "mini_infer/autograd.h"

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
    std::uint64_t input_versions[2] = {};
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class DivNode : public AutogradNode {
public:
    std::uint64_t input_versions[2] = {};
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

}  // namespace mtensor
