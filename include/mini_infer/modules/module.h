#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "mini_infer/tensor.h"

namespace mtensor {

// 只填初值；参数形状、设备和存储由层负责。
using WeightInitializer = std::function<void(float* values, std::size_t count)>;

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;

    Tensor operator()(const Tensor& x) { return forward(x); }

    // 按登记顺序返回参数句柄，和原参数共享数据及梯度。
    std::vector<Tensor> parameters() const {
        std::vector<Tensor> result;
        append_parameters(*this, result);
        return result;
    }

protected:
    // 由派生类的构造函数调用；不需要初始化参数的层可以沿用空实现。
    virtual void init() {}

    void register_parameter(const Tensor& parameter) { parameters_.push_back(parameter); }
    virtual void append_child_parameters(std::vector<Tensor>&) const {}

    // 组合层在遍历时传入现有子层，不保存成员地址。
    static void append_parameters(const Module& module, std::vector<Tensor>& result) {
        result.insert(result.end(), module.parameters_.begin(), module.parameters_.end());
        module.append_child_parameters(result);
    }

private:
    std::vector<Tensor> parameters_;
};

}  // namespace mtensor
