#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mini_infer/modules/module.h"

namespace {
using namespace mtensor;

class TestModule final : public Module {
public:
    TestModule(std::string name, float value)
        : Module(std::move(name)), weight_(full({}, value, Dtype::kFloat32)) {
        register_parameter("weight", weight_);
    }

    using Module::register_module;
    using Module::register_parameter;

    Tensor forward(const Tensor& x) override { return x; }
    const Tensor& weight() const { return weight_; }

private:
    Tensor weight_;
};

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckInvalidRegistrations() {
    TestModule root("Root", 1.0f);
    auto child = std::make_shared<TestModule>("Leaf", 2.0f);
    root.register_module("child", child);

    const auto expect_invalid = [](auto register_item, const std::string& message) {
        try {
            register_item();
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error(message);
    };
    for (const std::string name : {"", "a.b", "weight", "child"}) {
        expect_invalid([&] { root.register_parameter(name, root.weight()); },
                       "Accepted invalid parameter name '" + name + "'");
        expect_invalid([&] { root.register_module(name, child); },
                       "Accepted invalid child name '" + name + "'");
    }
    expect_invalid([&] { root.register_module<TestModule>("unused", nullptr); },
                   "Accepted a null child");
    const auto& children = root.named_children();
    const auto parameters = root.parameters();
    Check(children.size() == 1 && children[0].first == "child" && children[0].second == child &&
              parameters.size() == 2 && parameters[0].data_ptr() == root.weight().data_ptr() &&
              parameters[1].data_ptr() == child->weight().data_ptr(),
          "Rejected registrations must preserve the original child and parameters");
}

void CheckModuleTree() {
    TestModule root("Root", 1.0f);
    const auto left = std::make_shared<TestModule>("Branch", 2.0f);
    const auto leaf = std::make_shared<TestModule>("Leaf", 3.0f);
    const auto right = std::make_shared<TestModule>("Branch", 4.0f);
    root.register_module("left", left);
    left->register_module("leaf", leaf);
    root.register_module("right", right);
    right->register_module("shared_leaf", leaf);

    const auto& children = root.named_children();
    Check(children.size() == 2 && children[0].second == left && children[1].second == right,
          "named_children must contain only direct children in registration order");

    using Visits = std::vector<std::tuple<const Module*, std::string, std::size_t>>;
    Visits visits;
    const auto visit_fn = [&](const Module& module, std::string_view registered_name, std::size_t depth) {
        visits.emplace_back(&module, registered_name, depth);
    };
    std::unordered_set<const Module*> visited;
    root.visit_module_tree(visit_fn, visited);
    Check(visits == Visits{{&root, "", 0}, {left.get(), "left", 1},
                           {leaf.get(), "leaf", 2}, {right.get(), "right", 1}},
          "visit_module_tree must visit each module once at its first depth-first position");

    visits.clear();
    visited.clear();
    left->visit_module_tree(visit_fn, visited, "subtree", 3);
    Check(visits == Visits{{left.get(), "subtree", 3}, {leaf.get(), "leaf", 4}},
          "Traversal from a subtree must use the supplied root name and depth");

    const auto parameters = root.parameters();
    Check(parameters.size() == 4 &&
              parameters[0].data_ptr() == root.weight().data_ptr() &&
              parameters[1].data_ptr() == left->weight().data_ptr() &&
              parameters[2].data_ptr() == leaf->weight().data_ptr() &&
              parameters[3].data_ptr() == right->weight().data_ptr(),
          "parameters must return each module's original tensors before its children's");

    std::ostringstream out;
    root.print(out);
    Check(out.str() == "Root\n  left: Branch\n    leaf: Leaf\n  right: Branch\n",
          "print must show each module once under its first registered path");
}

}  // namespace

int main() {
    try {
        CheckModuleTree();
        CheckInvalidRegistrations();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
