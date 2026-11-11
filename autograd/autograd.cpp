#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

#include "autograd/autograd.h"

namespace mtensor{
namespace {
bool g_grad_enabled = true;

void TopoSort(
    std::unordered_set<AutogradNode*>& visited, 
    const Tensor& output, 
    std::vector<Tensor>& topo_order
){
    auto node = output.grad_node();
    if(!node) return ;
    if(visited.find(node.get()) != visited.end()){
        return ;
    }
    visited.insert(node.get());

    // 遍历输入
    for(const auto &input : node->inputs){
        TopoSort(visited, input, topo_order);
    }

    topo_order.push_back(output);
}

void CheckBackwardDtype(Dtype dtype) {
    if (dtype != Dtype::kFloat32 &&
        dtype != Dtype::kFloat16 &&
        dtype != Dtype::kBFloat16) {
        throw std::invalid_argument(
            "RunBackward: only floating-point dtypes are supported");
    }
}
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

void RunBackward(const Tensor& loss_tensor) {
    if(loss_tensor.empty() || loss_tensor.numel() != 1) { // loss 必须是个标量
        throw std::invalid_argument("RunBackward: loss_tensor must contain exactly one element");
    }

    CheckBackwardDtype(loss_tensor.dtype());

    if(!loss_tensor.grad_node()) {
        throw std::logic_error("RunBackward: loss_tensor does not have a grad node"); 
    }

    // TopoSort
    std::unordered_set<AutogradNode*> visited;
    std::vector<Tensor> topo_order;
    TopoSort(visited, loss_tensor, topo_order);
    std::reverse(topo_order.begin(), topo_order.end());

    // 将 g_grad_enabled 置为 false
    // 下列操作不建图
    NogradNode guard;   

    // 保存 "本次" backward 各节点的梯度，不能直接用 tensor 的 grad 历史梯度
    // 用作拓扑还没到 output 节点时临时保存 grad
    std::unordered_map<const AutogradNode*, Tensor> pending_grads;

    // loss 对自己的 grad 为 1
    Tensor loss_grad(loss_tensor.shape(), loss_tensor.dtype(), loss_tensor.device());
    loss_grad.fill_(1.0f);
    pending_grads.emplace(loss_tensor.grad_node().get(), loss_grad);

    // 反向遍历每个节点
    for(auto& output : topo_order){ // 第一个遍历的是 loss_tensor

        auto node = output.grad_node();
        Tensor grad_output = pending_grads.at(node.get());  // at 取自动报错
        if(output.requires_grad()){
            output.accumulate_grad(grad_output);
        }  

        // 根据输出梯度计算输入节点梯度
        std::vector<Tensor> input_grads = node->backward(grad_output);
        if(input_grads.size() != node->inputs.size()){
            throw std::logic_error("RunBackward: gradient count does not match input count");
        }

        for (auto i = 0; i < input_grads.size(); ++i){
            auto& input = node->inputs[i];
            const auto& input_grad = input_grads[i];
            auto input_node = input.grad_node();

            if(input_node){
                auto it = pending_grads.find(input_node.get());
                if(it == pending_grads.end()){  // 没有的话先添加
                    pending_grads.emplace(input_node.get(), input_grad);
                } else {
                    it->second = (it->second + input_grad);
                }
            } else if(input.requires_grad()) {  // 没有 autogradnode 但是需要保存梯度 -> 叶子节点
                input.accumulate_grad(input_grad);
            }
        }
    }
}
}
