#include <ostream>
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

void Module::visit_module_tree(const ModuleVisitor& visit_fn,
                               std::string_view registered_name, std::size_t depth) const {
    visit_fn(*this, registered_name, depth);
    for (const auto& [child_name, child] : children_) {
        child->visit_module_tree(visit_fn, child_name, depth + 1);
    }
}

void Module::print(std::ostream& out) const {
    visit_module_tree([&out](const Module& module, std::string_view registered_name, std::size_t depth) {
        out << std::string(depth * 2, ' ');
        if (!registered_name.empty()) out << registered_name << ": ";
        out << module.name() << '\n';
    });
}

std::vector<Tensor> Module::parameters() const {
    std::vector<Tensor> result;
    visit_module_tree([&result](const Module& module, std::string_view, std::size_t) {
        for (const auto& parameter : module.parameters_) {
            result.push_back(parameter.second);
        }
    });
    return result;
}

}  // namespace mtensor
