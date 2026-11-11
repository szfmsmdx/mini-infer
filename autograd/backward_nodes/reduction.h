#pragma once

#include <vector>

#include "autograd/autograd.h"

namespace mtensor {

// 全部元素求和的反向节点。
class SumNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

}  // namespace mtensor
