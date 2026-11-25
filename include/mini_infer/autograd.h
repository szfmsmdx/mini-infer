#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

#include "mini_infer/tensor/tensor.h"

namespace mtensor{
// 一个 Tensor 的自动微分状态；同一个 Tensor 的所有拷贝共享。
struct AutogradMetadata {
    bool requires_grad = false; // 是否保存该 Tensor 的梯度，不控制建图或梯度传递
    std::shared_ptr<AutogradNode> grad_node;   // 产生它的节点；叶子为空
    std::optional<Tensor> grad;                // 有值表示梯度已经写入
};

// 保存反向公式需要读取的前向 Tensor；读取前检查共享写入版本。
// 保存输出时先 detach，避免输出的 grad_node 经此对象指回自身。
class SavedTensor {
public:
    SavedTensor() = default;
    // is_output=true 时只保留数据和版本，不保留输出的 Autograd 历史。
    explicit SavedTensor(const Tensor& tensor, bool is_output = false);
    // 原地修改后抛 logic_error；成功时返回共享该数据的 Tensor 句柄。
    Tensor unpack(const char* debug_name = "SavedTensor") const;

private:
    std::optional<Tensor> tensor_;       // 保存的输入，或已 detach 的输出
    std::uint64_t saved_version_ = 0;    // 保存时的写入版本
    bool is_output_ = false;              // 错误消息区分前向输入与输出
};

// 反向图中的一个节点：一次前向调用对应一个节点。
class AutogradNode {
public:
    virtual ~AutogradNode() = default;

    // output_grad 与本次前向输出同 shape/dtype/device；
    // 返回值第 i 项是 inputs[i] 的梯度。
    virtual std::vector<Tensor> backward(const Tensor& output_grad) = 0;

    std::vector<Tensor> inputs;   // 本次前向的输入
    // 保存图的输入边，并记下每个输入在本次前向时是否需要梯度。
    void record_inputs(std::initializer_list<Tensor> values);
    // 手动设置 inputs 的旧节点没有掩码时，默认保留全部输入梯度。
    bool needs_input_grad(std::size_t index) const;

private:
    std::vector<bool> input_needs_grad_;  // 与 inputs 一一对应
};

// 检查当前 Autograd 是否支持该 dtype，支持时返回 true。
bool CheckGradDtype(Dtype dtype);
bool EnableGrad();
// 建图开启、输出 dtype 可求导，且至少一个输入要存梯度或连接到上游节点时为真。
bool ShouldRecordGrad(std::initializer_list<const Tensor*> inputs, Dtype output_dtype);

// 在当前线程临时关闭建图；离开作用域后恢复进入前的状态。
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    // 创建守卫前的建图开关状态；析构时恢复，支持守卫嵌套。
    bool previous_grad_enabled_;
};
using NogradNode = NoGradGuard;

void RunBackward(const Tensor& root);
void RunBackward(const Tensor& root, const Tensor& gradient);
}
