#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mini_infer/autograd.h"
#include "tensor/check.h"

namespace mtensor{
namespace {
thread_local bool g_grad_enabled = true;

void TopoSort(
    std::unordered_set<AutogradNode*>& visited, 
    const Tensor& output, 
    std::vector<Tensor>& topo_order
){
    auto node = output.grad_node();
    if (!node || !visited.insert(node.get()).second) {
        return;
    }

    // 遍历输入
    for(const auto &input : node->inputs){
        TopoSort(visited, input, topo_order);
    }

    topo_order.push_back(output);
}

}

bool CheckGradDtype(Dtype dtype) {
    return dtype == Dtype::kBFloat16 ||
           dtype == Dtype::kFloat16 ||
           dtype == Dtype::kFloat32;
}

bool EnableGrad() {
    return g_grad_enabled;
}

SavedTensor::SavedTensor(const Tensor& tensor, bool is_output)
    : tensor_(is_output ? tensor.detach() : tensor),
      saved_version_(tensor.version()), is_output_(is_output) {}

Tensor SavedTensor::unpack(const char* debug_name) const {
    if (!tensor_) throw std::logic_error(std::string(debug_name) + ": no saved tensor");
    if (tensor_->version() != saved_version_) {
        throw std::logic_error(std::string(debug_name) +
            (is_output_ ? ": output modified after forward" : ": input modified after forward"));
    }
    return *tensor_;
}

void AutogradNode::record_inputs(std::initializer_list<Tensor> values) {
    inputs.assign(values);
    input_needs_grad_.clear();
    input_needs_grad_.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        input_needs_grad_.push_back(input.requires_grad() || input.grad_node() != nullptr);
    }
}

bool AutogradNode::needs_input_grad(std::size_t index) const {
    if (index >= inputs.size()) throw std::out_of_range("AutogradNode: input index out of range");
    return input_needs_grad_.empty() || input_needs_grad_[index];
}

bool ShouldRecordGrad(std::initializer_list<const Tensor*> inputs, Dtype output_dtype) {
    if (!EnableGrad() || !CheckGradDtype(output_dtype)) return false;
    for (const Tensor* input : inputs) {
        if (input && (input->requires_grad() || input->grad_node())) return true;
    }
    return false;
}

NoGradGuard::NoGradGuard() : previous_grad_enabled_(g_grad_enabled) {
    g_grad_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    g_grad_enabled = previous_grad_enabled_;
}

void RunBackward(const Tensor& loss_tensor) {
    if(loss_tensor.empty() || loss_tensor.numel() != 1) { // loss 必须是个标量
        throw std::invalid_argument("RunBackward: loss_tensor must contain exactly one element");
    }

    CheckFloatingDtype(loss_tensor.dtype(), "RunBackward: only floating-point dtypes are supported");

    if(!loss_tensor.grad_node()) {
        throw std::logic_error("RunBackward: loss_tensor does not have a grad node"); 
    }

    Tensor gradient = full(loss_tensor.shape(), 1, loss_tensor.dtype(), loss_tensor.device());
    RunBackward(loss_tensor, gradient);
}

void RunBackward(const Tensor& root, const Tensor& gradient) {
    CheckNotEmpty(root, "RunBackward: root must not be empty");
    CheckNotEmpty(gradient, "RunBackward: gradient must not be empty");
    CheckFloatingDtype(root.dtype(), "RunBackward: only floating-point dtypes are supported");
    CheckSameShape(root, gradient, "RunBackward: gradient shape must match root");
    CheckSameDtype(root, gradient, "RunBackward: gradient dtype must match root");
    CheckSameDevice(root, gradient, "RunBackward: gradient device must match root");
    const auto root_node = root.grad_node();
    if (!root_node) {
        throw std::logic_error("RunBackward: root does not have a grad node");
    }

    // TopoSort
    std::unordered_set<AutogradNode*> visited;
    std::vector<Tensor> topo_order;
    TopoSort(visited, root, topo_order);
    std::reverse(topo_order.begin(), topo_order.end());

    // 将 g_grad_enabled 置为 false
    // 下列操作不建图
    NogradNode guard;   

    // 保存 "本次" backward 各节点的梯度，不能直接用 tensor 的 grad 历史梯度
    // 用作拓扑还没到 output 节点时临时保存 grad
    std::unordered_map<const AutogradNode*, Tensor> pending_grads;

    // 独立保存调用方传入的上游梯度，不与外部梯度存储共享。
    pending_grads.emplace(root_node.get(), gradient.clone());

    // 反向遍历每个节点
    for(auto& output : topo_order){ // 第一个遍历的是 root

        auto node = output.grad_node();
        Tensor output_grad = pending_grads.at(node.get());  // at 取自动报错
        if(output.requires_grad()){
            output.accumulate_grad(output_grad);
        }  

        // 根据输出梯度计算输入节点梯度
        std::vector<Tensor> input_grads = node->backward(output_grad);
        if(input_grads.size() != node->inputs.size()){
            throw std::logic_error("RunBackward: gradient count does not match input count");
        }

        for (auto i = 0; i < input_grads.size(); ++i){
            if (!node->needs_input_grad(i)) continue;
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
