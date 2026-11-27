#include <stdexcept>
#include <utility>

#include "mini_infer/modules/module.h"

namespace mtensor {

void Module::check_registration_name(const std::string& name) const {
    if (name.empty() || name.find('.') != std::string::npos) {
        throw std::invalid_argument("Module: registration name must be nonempty and contain no dots");
    }
    for (const auto& parameter : parameters_) {
        if (parameter.first == name) {
            throw std::invalid_argument("Module: name already registered: " + name);
        }
    }
    for (const auto& child : children_) {
        if (child.first == name) {
            throw std::invalid_argument("Module: name already registered: " + name);
        }
    }
}

void Module::register_parameter(std::string name, const Tensor& parameter) {
    check_registration_name(name);
    parameters_.emplace_back(std::move(name), parameter);
}

void Module::register_child(std::string name, std::shared_ptr<Module> child) {
    check_registration_name(name);
    if (!child) throw std::invalid_argument("Module: child module must not be null");
    children_.emplace_back(std::move(name), std::move(child));
}

void Module::collect_parameters(std::vector<Tensor>& result) const {
    // 先收集本模块的参数，再按登记顺序遍历子模块。
    for (const auto& parameter : parameters_) {
        result.push_back(parameter.second);
    }
    for (const auto& child : children_) {
        child.second->collect_parameters(result);
    }
}

std::vector<Tensor> Module::parameters() const {
    std::vector<Tensor> result;
    collect_parameters(result);
    return result;
}

}  // namespace mtensor
