#pragma once

#include <memory>
#include <vector>
#include <optional>

#include "mini_infer/core/tensor.h"

namespace mtensor{
// 一个 Tensor 的自动微分状态；同一个 Tensor 的所有拷贝共享。
struct AutogradMetadata {
    bool requires_grad = false; // 是否保存该 Tensor 的梯度，不控制建图或梯度传递
    std::shared_ptr<AutogradNode> grad_node;   // 产生它的节点；叶子为空
    std::optional<Tensor> grad;                // 有值表示梯度已经写入
};

// 反向图中的一个节点：一次前向调用对应一个节点。
class AutogradNode {
public:
    virtual ~AutogradNode() = default;

    // output_grad 与本次前向输出同 shape/dtype/device；
    // 返回值第 i 项是 inputs[i] 的梯度。
    virtual std::vector<Tensor> backward(const Tensor& output_grad) = 0;

    std::vector<Tensor> inputs;   // 本次前向的输入
};

// 检查当前 Autograd 是否支持该 dtype，支持时返回 true。
bool CheckGradDtype(Dtype dtype);
bool EnableGrad();

class NogradNode{
public:
    NogradNode();
    ~NogradNode();

    NogradNode(const NogradNode&) = delete;
    NogradNode& operator=(const NogradNode&) = delete;

private:
    // 创建守卫前的建图开关状态；析构时恢复，支持守卫嵌套。
    bool previous_grad_enabled_;
};

void RunBackward(const Tensor& root);
void RunBackward(const Tensor& root, const Tensor& gradient);
}
