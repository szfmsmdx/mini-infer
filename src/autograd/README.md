# autograd 设计

支持训练的算子在前向时挂上反向节点，`loss.backward()` 沿计算图传递梯度。`NogradNode` 关闭建图，`requires_grad` 决定是否保存该 Tensor 的梯度。

## 对象关系

```text
Tensor ──持有──→ AutogradMetadata
                    ├─ bool requires_grad
                    ├─ std::optional<Tensor> grad
                    └─ grad_node ──→ AutogradNode
                                      ├─ inputs：std::vector<Tensor>（本次前向的输入）
                                      └─ backward(output_grad)
```

- `AutogradMetadata` 保存一个 Tensor 的全部自动微分状态。
- `Tensor` 用 `std::shared_ptr<AutogradMetadata>` 持有它；Tensor 拷贝共享同一个 meta，视图创建新的 meta。
- `meta.grad_node` 指向**产生这个 Tensor 的 Node**；叶子张量的 `grad_node` 为空。
- Node 用 `inputs` 连接前向输入。`ExpNode` 还保存 `y.detach()` 和输出版本号：它共享输出数据，但不持有指回节点的 meta。
- 每个输入 Tensor 又有自己的 meta 和 `grad_node`，图按这个方向串联。

## 文件结构

| 文件或目录 | 职责 |
| --- | --- |
| `include/mini_infer/autograd.h` | 元数据、节点接口和梯度开关 |
| `src/autograd/autograd.cpp` | 拓扑排序和梯度累加 |
| `src/ops/` | 逐元素、矩阵乘、归约、视图、RMSNorm、词嵌入、RoPE、因果注意力和交叉熵的节点 |

[算子说明](../ops/README.md) · [精度排查](../../note/code/算子精度问题.md)

## 数据结构

```cpp
// include/mini_infer/tensor.h
std::shared_ptr<AutogradMetadata> autograd_metadata_;

// include/mini_infer/autograd.h
struct AutogradMetadata {
    bool requires_grad = false;
    std::shared_ptr<AutogradNode> grad_node;   // 产生它的节点；叶子为空
    std::optional<Tensor> grad;              // 有值表示梯度已经写入
};

class AutogradNode {
public:
    virtual ~AutogradNode() = default;
    // output_grad 与本次前向输出同 shape/dtype/device；
    // 返回值第 i 项对应 inputs[i]。
    virtual std::vector<Tensor> backward(const Tensor& output_grad) = 0;

    std::vector<Tensor> inputs;   // 本次前向的输入，同时延长其存储生命周期
};
```

- `grad` 用 `std::optional<Tensor>`：默认不构造 Tensor，避免“Tensor 构造 meta、meta 再构造 Tensor”的递归；`has_value()` 直接表示梯度是否存在。
- 普通构造创建空 meta。拷贝共享它，因此先拷贝再调用 `requires_grad_` 也会更新同一份状态。视图各自创建 meta。
- 引擎遍历时临时持有输出 Tensor 句柄，并从本次反向的临时梯度表读取输出梯度。

## Tensor 接口

```cpp
bool requires_grad() const;
Tensor& requires_grad_(bool value);   // 控制是否保存梯度
bool has_grad() const;                // meta->grad.has_value()
Tensor grad() const;                  // 无梯度返回默认空 Tensor
void zero_grad();                     // 已有梯度清零，保留存储和 has_grad 状态
void backward();                      // 单元素入口，默认上游梯度为 1
void backward(const Tensor& gradient); // 指定上游梯度，须同 shape/dtype/device
Tensor detach() const;                // 共享存储，使用新的空 meta

// 框架内部
std::shared_ptr<AutogradNode> grad_node() const;   // meta ? meta->grad_node : nullptr
void set_grad_node(std::shared_ptr<AutogradNode> node);
void accumulate_grad(const Tensor& grad);
```

`set_grad_node()` 只设置 `meta->grad_node`，不修改 `requires_grad`；叶子和非叶子都可以设置是否保存梯度。

`accumulate_grad()` 首次克隆梯度，之后用 `add(*meta->grad, grad)` 累加。

## 前向建图

以 add 为例：

```cpp
Tensor add(const Tensor& a, const Tensor& b) {
    // 现有检查、计算 ...
    if (EnableGrad() && /* 本次运算支持反向的浮点 dtype */) {
        auto node = std::make_shared<AddNode>();
        node->inputs = {a, b};
        out.set_grad_node(std::move(node));
    }
    return out;
}

class AddNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override { return {output_grad, output_grad}; }
};
```

启用建图时，支持反向的浮点运算创建节点。输入的 `requires_grad` 控制梯度保存。整数运算直接返回计算结果。推理时用 `NogradNode` 关闭建图，`EnableGrad()` 可查询当前状态。

## 反向引擎

```text
RunBackward(root, gradient)
1. 检查 root 与 gradient 非空、同 shape/dtype/device，root 为浮点且 grad_node 非空；无参入口额外要求 root.numel() == 1
2. TopoSort(root)：DFS 后序，按 node 指针去重，在 topo_order 中保存输出 Tensor
     通过输出的 grad_node() 取得 Node；将前向拓扑序反转为反向执行顺序
3. 创建 NogradNode：后续梯度计算和累加不再建图
4. 为 root 对应的节点写入 gradient.clone()；无参入口创建全 1 梯度后复用此流程
5. 逆序处理每一项：
     output_grad = 临时梯度表中该节点的梯度
     输出 Tensor 的 requires_grad 为 true 时，先累加到持久 grad（此刻梯度已收齐）
     input_grads = node->backward(output_grad)
     逐个输入：有 grad_node 的累加到临时表；叶子若 requires_grad 则直接累加到持久 grad
```

每次 `backward()` 新建临时梯度表。多次反向会累加保存在 Tensor 上的梯度。用 `zero_grad()` 清零已有梯度，再开始下一轮训练。

## 算子反向公式

| 算子 | 输入梯度 |
| --- | --- |
| `add(a, b)` | `dA = G`, `dB = G` |
| `sub(a, b)` | `dA = G`, `dB = -G` |
| `mul(a, b)` | `dA = G * b`, `dB = G * a` |
| `div(a, b)` | `dA = G / b`, `dB = -G * a / (b * b)` |
| `neg(a)` | `dA = -G` |

`G` 是上游梯度，由引擎传入。`add` / `sub` / `mul` / `div` 自动广播时，反向节点保存广播后的输入，上述公式产生公共 shape 的梯度，再由 `BroadcastNode` 沿广播维度归约到原输入 shape。

## 原地修改检测

Tensor 通过共享的 `version_` 记录存储修改次数。拷贝、共享视图和 `detach()` 共用计数器；`clone()`、跨设备复制等独立存储使用新计数器。

`fill_`（普通数值和标量 Tensor）、`zero_`、`copy_` 在校验完成并开始写入前调用 `update_version()`，`NogradNode` 下也更新。空操作、同位置的 `copy_` 和校验失败都保持原版本。开始写入后若失败，保留已递增的版本。`zero_grad()` 更新梯度存储的版本。

`MulNode`、`DivNode` 在前向保存两个输入的版本，反向读取输入数值前通过 `CheckVersion` 检查，不一致时抛 `std::logic_error`。输入是否请求保存梯度不影响检查；`add/sub/neg` 及当前视图、归约反向不读取原输入数值，不检查数据版本。

通过 `data_ptr()` 或自定义 kernel 写内存前，调用方须调用 `update_version()`。反向读取保存的 Tensor 数值时，节点应保存并检查对应的版本。

## 接口约束

- 无参 `backward()` 只支持单元素输出；`backward(gradient)` 支持非标量输出，gradient 须非空且与输出同 shape/dtype/device。两个入口均要求输出有 grad_node。
- 支持 Float32、Float16、BFloat16 梯度；`RunBackward` 拒绝整数 dtype，`requires_grad_` 本身不限制 dtype。
- `operator[]`、`reshape()`、`narrow()`、`transpose()`、`permute()`、`clone()` 和跨设备 `to()` 接入反向节点。
- `zero_` / `fill_` / `copy_` 不建图；优化器更新需在 `NogradNode` 下进行。
- `detach()` 共享存储但使用新的空 meta，其梯度不会回传到原 Tensor。

[Autograd 测试](../../tests/README.md#测试覆盖)
