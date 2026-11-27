#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

// 只填初值；参数形状、设备和存储由层负责。
using WeightInitializer = std::function<void(float* values, std::size_t count)>;

class Module {
public:
    using NamedChild = std::pair<std::string, std::shared_ptr<Module>>;

    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;

    Tensor operator()(const Tensor& x) { return forward(x); }

    // 只返回直接子模块，不包含参数。
    const std::vector<NamedChild>& named_children() const noexcept { return children_; }

    // 按登记顺序递归收集；返回的句柄与原参数共享数据和梯度。
    std::vector<Tensor> parameters() const;

protected:
    // 由派生类的构造函数调用；不需要初始化参数的层可以沿用空实现。
    virtual void init() {}

    void register_parameter(std::string name, const Tensor& parameter);

    template <typename T>
    std::shared_ptr<T> register_module(std::string name, std::shared_ptr<T> child) {
        register_child(std::move(name), child);
        return child;
    }

private:
    void register_child(std::string name, std::shared_ptr<Module> child);
    void check_registration_name(const std::string& name) const;
    void collect_parameters(std::vector<Tensor>& result) const;

    std::vector<std::pair<std::string, Tensor>> parameters_;
    std::vector<NamedChild> children_;
};

}  // namespace mtensor
