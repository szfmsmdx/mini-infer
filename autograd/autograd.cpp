#include <stdexcept>
#include <unordered_set>

#include "autograd/autograd.h"

namespace mtensor{
namespace {
bool g_grad_enabled = true;
}

bool EnableGrad() {
    return g_grad_enabled;
}

NogradNode::NogradNode() : previous_grad_enabled_(g_grad_enabled) {
    g_grad_enabled = false;
}

NogradNode::~NogradNode() {
    g_grad_enabled = previous_grad_enabled_;
}

void RunBackward(const Tensor& root) {
    if(root.empty() || root.numel() != 1) { // loss 必须是个标量
        throw std::invalid_argument("RunBackward: root must contain exactly one element");
    }

    if(!root.grad_node()) {
        throw std::logic_error("RunBackward: root does not have a grad node"); 
    }

    // 防止反向传播中的 node 继续建图
    NogradNode guard;

    std::unordered_set<AutogradNode*> visted;

}
}
