#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
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

    Tensor forward(const Tensor& x) override { return x; }
    const Tensor& weight() const { return weight_; }

private:
    Tensor weight_;
};

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckModuleTree() {
    TestModule root("Root", 1.0f);
    const auto left = std::make_shared<TestModule>("Branch", 2.0f);
    const auto leaf = std::make_shared<TestModule>("Leaf", 3.0f);
    const auto right = std::make_shared<TestModule>("Branch", 4.0f);
    root.register_module("left", left);
    left->register_module("leaf", leaf);
    root.register_module("right", right);

    const auto& children = root.named_children();
    Check(children.size() == 2 && children[0].second == left && children[1].second == right,
          "named_children must contain only direct children in registration order");

    using Visits = std::vector<std::tuple<const Module*, std::string, std::size_t>>;
    Visits visits;
    const auto visit_fn = [&](const Module& module, std::string_view registered_name, std::size_t depth) {
        visits.emplace_back(&module, registered_name, depth);
    };
    root.visit_module_tree(visit_fn);
    Check(visits == Visits{{&root, "", 0}, {left.get(), "left", 1},
                           {leaf.get(), "leaf", 2}, {right.get(), "right", 1}},
          "visit_module_tree must visit the root then each subtree in depth-first order");

    visits.clear();
    left->visit_module_tree(visit_fn, "subtree", 3);
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
          "print must preserve the tree structure and distinguish registered names from module names");
}

}  // namespace

int main() {
    try {
        CheckModuleTree();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
