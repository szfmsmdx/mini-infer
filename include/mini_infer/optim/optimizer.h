#pragma once

#include <vector>

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    // 优化器持有共享参数句柄，优化器自身的状态随对象移动。
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    Optimizer(Optimizer&&) noexcept = default;
    Optimizer& operator=(Optimizer&&) noexcept = default;

    void zero_grad();
    virtual void step() = 0;

protected:
    Optimizer(std::vector<Tensor> parameters, float learning_rate);
    void ValidateGradients() const;

    std::vector<Tensor> parameters_;
    float learning_rate_;
};

}  // namespace mtensor
