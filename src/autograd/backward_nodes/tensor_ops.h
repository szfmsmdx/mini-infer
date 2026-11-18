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

// inputs[0] 保存前向输入，原 shape/dtype/device 从中读取。
class ReshapeNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class NarrowNode : public AutogradNode {
public:
    int dim = 0;
    int start = 0;
    int length = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

// operator[] 只沿第 0 维索引，并移除该维度。
class IndexNode : public AutogradNode {
public:
    int index = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SqueezeNode : public AutogradNode {
public:
    // 保存实际移除的维度；未移除维度时不使用此节点。
    int dim = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class UnsqueezeNode : public AutogradNode {
public:
    // 保存实际插入的维度。
    int dim = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

}  // namespace mtensor
