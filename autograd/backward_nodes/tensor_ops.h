#pragma once

#include <vector>

#include "autograd/autograd.h"

namespace mtensor {

// clone() 的反向节点：将输出梯度传给输入。
class CloneNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

// 跨设备 to() 的反向节点：将输出梯度复制回输入设备。
class ToDeviceNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

// transpose() 的反向节点。
class TransposeNode : public AutogradNode {
public:
    int dim0, dim1;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

// permute() 的反向节点。
class PermuteNode : public AutogradNode {
public:
    std::vector<int> dims;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

}  // namespace mtensor
