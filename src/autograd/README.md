# autograd 设计

**目标：同一套 forward 同时支持训练和推理。** 训练时前向自动建图，`loss.backward()` 计算梯度；推理时关闭建图。Module 只写 `forward`，基础算子的反向由每次前向创建的 Node 负责。这里的 `requires_grad` 与 PyTorch 含义不同：它只决定是否保存该 Tensor 的梯度，不决定是否建图或继续传递梯度。

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

一句话：**Meta 存状态，Tensor 持有 Meta，Meta 指向产生它的 Node，Node 持有输入 Tensor。**

## 文件结构

```text
include/mini_infer/
    autograd.h           # 元数据、节点基类、梯度开关和反向入口

src/autograd/
    autograd.cpp    # 梯度开关与反向引擎
    backward_nodes/
        elementwise.h    # 逐元素反向节点声明
        elementwise.cpp  # 逐元素反向规则与 sigmoid/rsqrt/SiLU CPU 反向
        elementwise.cu   # sigmoid/rsqrt/SiLU CUDA 反向
        reduction.h      # 归约反向节点声明
        reduction.cpp    # 归约反向节点（SumNode）
        tensor_ops.h     # 复制、迁移与视图节点声明
        tensor_ops.cpp   # 复制、迁移与视图的反向规则
```

逐元素 Node 在 `src/autograd/backward_nodes/elementwise.h/.cpp`，前向入口在 `src/ops/elementwise.cpp` 中创建节点。反向规则按算子类别组织。sigmoid 的 CPU 反向放在同一个 `.cpp`，CUDA kernel 放在相邻的 `elementwise.cu`。两者从输入重新计算 `t = exp(-abs(x))`，在 FP32 中计算 `(G * t) / ((1 + t) * (1 + t))`，最后写回原 dtype；不使用已舍入的前向输出计算导数。原有输入、输出版本检查保留。

rsqrt 也在这两个文件中逐元素计算反向：从输入在 FP32 中重新计算 `y = 1 / sqrt(x)`，依次计算 `((-0.5 * y) * G) * y * y`，最后才写回原 dtype。保留输入版本检查，CPU/CUDA 均按各自 stride 读取输入和上游梯度。

SiLU 反向也在这两个文件中执行，采用 Torch 的运算顺序：FP32 计算 `s = 1 / (1 + exp(-x))`，然后计算 `(G * s) * (1 + x * (1 - s))`，最后写回原 dtype。不分 x 区间、不使用 double，保留输入版本检查。

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
- 每个 Tensor 构造时创建一个空 meta；拷贝共享同一个 `shared_ptr`，视图各自新建。这样先拷贝再 `requires_grad_` 也能共享状态。代价是每个 Tensor 多一次小分配；以后若改成共享 `TensorImpl`，把 `AutogradMetadata` 移进去即可，名字和字段不变。
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

`accumulate_grad()` 第一次执行 `meta->grad.emplace(grad.clone())`，之后用 `add(*meta->grad, grad)` 累加，避免多个输入共享同一块梯度存储。

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

公共算子接口不变。建图开关开启时，支持反向的浮点运算创建节点，与输入是否保存梯度无关；整数运算不建反向图。推理可用守卫关闭建图。代码中的查询函数目前叫 `EnableGrad()`，守卫叫 `NogradNode`。

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

临时梯度表以 `const AutogradNode*` 为键：每个前向调用创建唯一的 Node，就是该输出的身份；没有 `grad_node` 的 Tensor 是叶子，梯度不再上传，直接写持久梯度，因此不需要给 Tensor 增加 meta 访问器。中间量的持久梯度在它自己的节点被处理时写入，不会重复累加。临时梯度与持久 `grad` 分开，避免重复 `backward()` 时把旧梯度再次用于传递。第一次实现不释放计算图，多次 `backward()` 会继续累加持久梯度，通过 Tensor::zero_grad() 清零已保存的梯度；仅作用于当前 Tensor，不遍历计算图，不改变 requires_grad 或 grad_node。

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

`fill_`（普通数值和标量 Tensor）、`zero_`、`copy_` 在校验完成、实际写入前调用 `update_version()`，`NogradNode` 下也更新。空操作、同位置的 `copy_` 和参数校验失败不更新；写入开始后若失败，版本保守地保持已递增状态。`zero_grad()` 修改梯度存储的版本，不改变参数存储的版本。

`MulNode`、`DivNode` 在前向保存两个输入的版本，反向读取输入数值前通过 `CheckVersion` 检查，不一致时抛 `std::logic_error`。输入是否请求保存梯度不影响检查；`add/sub/neg` 及当前视图、归约反向不读取原输入数值，不检查数据版本。

通过 `data_ptr()` 或自定义 kernel 写内存时，框架无法自动感知，调用方须在写入前显式调用 `update_version()`。新算子的反向如果依赖保存的 Tensor 数值，也需要保存并检查对应版本。

## 第一阶段限制

- 无参 `backward()` 只支持单元素输出；`backward(gradient)` 支持非标量输出，gradient 须非空且与输出同 shape/dtype/device。两个入口均要求输出有 grad_node。
- 支持 Float32、Float16、BFloat16 梯度；`RunBackward` 拒绝整数 dtype，`requires_grad_` 本身不限制 dtype。
- `operator[]`、`reshape()`、`narrow()`、`transpose()`、`permute()`、`clone()` 和跨设备 `to()` 接入反向节点。`detach()` 和同设备 `to()` 不受影响。
- `zero_` / `fill_` / `copy_` 不建图；优化器更新需在 `NogradNode` 下进行。
- `detach()` 共享存储但使用新的空 meta，其梯度不会回传到原 Tensor。

## 任务清单

- [x] 梯度元数据、建图开关和 `NogradNode` 守卫。
- [x] 反向拓扑排序、临时梯度汇总和持久梯度累加。
- [x] 逐元素算子、`clone`、`transpose`、`permute` 和全元素 `sum` 的反向规则。
- [x] 跨设备 `to()` 的反向节点与 CUDA 测试代码。
- [x] 在 Colab A100 验证跨设备迁移、转置视图和 `sum` 的相关前向与反向测试。
- [x] 支持连续及非连续输入的 `reshape` 反向；非连续上游梯度先 clone，再恢复输入 shape。本地 CPU 及 Colab CPU/CUDA 回归已通过。
- [x] 支持 `narrow` 反向和前向建图，本地 CPU 及 Colab CPU/CUDA 回归已通过。
- [x] 支持 `operator[]` 的视图反向与前向建图，本地 CPU 及 Colab CPU/CUDA 回归已通过。
- [x] 二维矩阵乘反向复用转置视图与现有 matmul；反向前检查两个输入的版本。
- [x] 支持指定维度的 `sum` 及广播视图的反向。
- [x] 支持非标量输出的 `backward(gradient)`；本地 CPU 回归 8/8 通过，用户在 Colab 完成 CPU/CUDA 回归，CTest 13/13 通过（2.30 秒）。
- [x] Tensor::zero_grad() 保留梯度存储并清零；本地 CPU 及用户 Colab CPU/CUDA 回归已通过。
- [x] 共享版本计数、原地写入更新及 mul/div 反向版本检查；本地 CPU 及用户 Colab CPU/CUDA 回归已通过。
