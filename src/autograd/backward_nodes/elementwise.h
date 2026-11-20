#pragma once

#include <cstdint>

#include "mini_infer/autograd.h"

namespace mtensor {

// 内部 CUDA 后端：按各自 stride 读取 x 和上游梯度，写入连续 dx。
void SigmoidBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);
void RsqrtBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);
void SiluBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx);

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

class SqrtNode : public AutogradNode {
public:
    std::uint64_t input_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class RsqrtNode : public AutogradNode {
public:
    std::uint64_t input_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class ExpNode : public AutogradNode {
public:
    Tensor y;
    std::uint64_t output_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SigmoidNode : public AutogradNode {
public:
    Tensor y;
    std::uint64_t input_version = 0;
    std::uint64_t output_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SiluNode : public AutogradNode {
public:
    std::uint64_t input_version = 0;
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
