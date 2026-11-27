#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mini_infer/tensor/tensor.h"

namespace mtensor {

// 只填初值；参数形状、设备和存储由层负责。
using WeightInitializer = std::function<void(float* values, std::size_t count)>;

class Module {
public:
    using NamedChild = std::pair<std::string, std::shared_ptr<Module>>;
    using ModuleVisitor = std::function<void(const Module&, std::string_view registered_name, std::size_t depth)>;

    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;

    Tensor operator()(const Tensor& x) { return forward(x); }
    const std::string& name() const noexcept { return name_; }

    // 只返回直接子模块，不包含参数。
    const std::vector<NamedChild>& named_children() const noexcept { return children_; }

    // 每次遍历传入空集合；按登记顺序深度优先访问每个模块一次。
    void visit_module_tree(const ModuleVisitor& visit_fn, std::unordered_set<const Module*>& visited,
                           std::string_view registered_name = {}, std::size_t depth = 0) const;
    void print(std::ostream& out) const;

    // 按登记顺序递归收集；返回的句柄与原参数共享数据和梯度。
    std::vector<Tensor> parameters() const;

protected:
    explicit Module(std::string name = "Module") : name_(std::move(name)) {}

    // 由派生类的构造函数调用；不需要初始化参数的层可以沿用空实现。
    virtual void init() {}

    void register_parameter(std::string name, const Tensor& parameter);

    // 可以共享子模块，调用方须保证登记关系无环。
    template <typename T>
    std::shared_ptr<T> register_module(std::string name, std::shared_ptr<T> child) {
        register_child(std::move(name), child);
        return child;
    }

private:
    void register_child(std::string name, std::shared_ptr<Module> child);
    void check_registration_name(const std::string& name) const;

    std::string name_;
    std::vector<std::pair<std::string, Tensor>> parameters_;
    std::vector<NamedChild> children_;
};

}  // namespace mtensor
