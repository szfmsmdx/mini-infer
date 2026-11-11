# autograd 设计

**目标：同一套 forward 同时支持训练和推理。** 训练时前向自动建图，`loss.backward()` 计算梯度；推理时关闭建图。Module 只写 `forward`，基础算子的反向由每次前向创建的 Node 负责。这里的 `requires_grad` 与 PyTorch 含义不同：它只决定是否保存该 Tensor 的梯度，不决定是否建图或继续传递梯度。

## 对象关系

```text
Tensor ──持有──→ AutogradMetadata
                    ├─ bool requires_grad
                    ├─ std::optional<Tensor> grad
                    └─ grad_node ──→ AutogradNode
                                      ├─ inputs：std::vector<Tensor>（本次前向的输入）
                                      └─ backward(grad_output)
```

- `AutogradMetadata` 保存一个 Tensor 的全部自动微分状态。
- `Tensor` 用 `std::shared_ptr<AutogradMetadata>` 持有它；Tensor 拷贝共享同一个 meta，视图创建新的 meta。
- `meta.grad_node` 指向**产生这个 Tensor 的 Node**；叶子张量的 `grad_node` 为空。
- Node 只保存**输入 Tensor**，不保存输出；因此没有 `Node → 输出` 的强引用，不会形成环。
- 每个输入 Tensor 又有自己的 meta 和 `grad_node`，图按这个方向串联。

一句话：**Meta 存状态，Tensor 持有 Meta，Meta 指向产生它的 Node，Node 持有输入 Tensor。**

## 文件结构

```text
autograd/
    autograd.h      # AutogradMetadata、AutogradNode、EnableGrad、NogradNode、RunBackward
    autograd.cpp    # 梯度开关与反向引擎
    README.md
```

基础算子的 Node 不单独建文件，放在对应算子实现文件的匿名命名空间里。当前只有 `ops/elementwise.cpp`；后续 `matmul`、`softmax` 等按同样方式处理。

## 数据结构

```cpp
// tensor/tensor.h
std::shared_ptr<AutogradMetadata> autograd_metadata_;

// autograd/autograd.h
struct AutogradMetadata {
    bool requires_grad = false;
    std::shared_ptr<AutogradNode> grad_node;   // 产生它的节点；叶子为空
    std::optional<Tensor> grad;              // 有值表示梯度已经写入
};

class AutogradNode {
public:
    virtual ~AutogradNode() = default;
    // grad_output 与本次前向输出同 shape/dtype/device；
    // 返回值第 i 项对应 inputs[i]。
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;

    std::vector<Tensor> inputs;   // 本次前向的输入，同时延长其存储生命周期
};
```

- `grad` 用 `std::optional<Tensor>`：默认不构造 Tensor，避免“Tensor 构造 meta、meta 再构造 Tensor”的递归；`has_value()` 直接表示梯度是否存在。
- 每个 Tensor 构造时创建一个空 meta；拷贝共享同一个 `shared_ptr`，视图各自新建。这样先拷贝再 `requires_grad_` 也能共享状态。代价是每个 Tensor 多一次小分配；以后若改成共享 `TensorImpl`，把 `AutogradMetadata` 移进去即可，名字和字段不变。
- Node 不保存输出或输出梯度。引擎遍历时临时持有输出 Tensor 句柄，并从本次反向的临时梯度表读取梯度。

## Tensor 接口

```cpp
bool requires_grad() const;
Tensor& requires_grad_(bool value);   // 控制是否保存梯度
bool has_grad() const;                // meta->grad.has_value()
Tensor grad() const;                  // 无梯度返回默认空 Tensor
void backward();                      // 标量入口，调用 RunBackward
Tensor detach() const;                // 共享存储，使用新的空 meta

// 框架内部
std::shared_ptr<AutogradNode> grad_node() const;   // meta ? meta->grad_node : nullptr
void set_grad_node(std::shared_ptr<AutogradNode> node);
void accumulate_grad(const Tensor& grad);
```

`set_grad_node()` 只设置 `meta->grad_node`，不修改 `requires_grad`；叶子和非叶子都可以设置是否保存梯度。Node 不保存输出，也不保存输出梯度。

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
    std::vector<Tensor> backward(const Tensor& g) override { return {g, g}; }
};
```

公共算子接口不变。建图开关开启时，支持反向的浮点运算创建节点，与输入是否保存梯度无关；整数运算不建反向图。推理可用守卫关闭建图。代码中的查询函数目前叫 `EnableGrad()`，守卫叫 `NogradNode`。

## 反向引擎

```text
RunBackward(root)
1. 检查 root 非空、numel() == 1、grad_node 非空
2. 创建 NogradNode：反向中的 add/mul 不再建图
3. TopoSort(root)：DFS 后序，按 node 指针去重，记录 {node, 产生它的输出 Tensor}
     order 是前向拓扑序（生产者在前），反向从末尾往开头遍历
4. 为 root 对应的节点写入全 1 的 seed
5. 逆序处理每一项：
     grad_output = 临时梯度表中该节点的梯度
     输出 Tensor 的 requires_grad 为 true 时，先累加到持久 grad（此刻梯度已收齐）
     input_grads = node->backward(grad_output)
     逐个输入：有 grad_node 的累加到临时表；叶子若 requires_grad 则直接累加到持久 grad
```

临时梯度表以 `const AutogradNode*` 为键：每个前向调用创建唯一的 Node，就是该输出的身份；没有 `grad_node` 的 Tensor 是叶子，梯度不再上传，直接写持久梯度，因此不需要给 Tensor 增加 meta 访问器。中间量的持久梯度在它自己的节点被处理时写入，不会重复累加。临时梯度与持久 `grad` 分开，避免重复 `backward()` 时把旧梯度再次用于传递。第一次实现不释放计算图，多次 `backward()` 会继续累加持久梯度，清梯度留到优化器阶段。

## 算子反向公式

| 算子 | 输入梯度 |
| --- | --- |
| `add(a, b)` | `dA = G`, `dB = G` |
| `sub(a, b)` | `dA = G`, `dB = -G` |
| `mul(a, b)` | `dA = G * b`, `dB = G * a` |
| `div(a, b)` | `dA = G / b`, `dB = -G * a / (b * b)` |
| `neg(a)` | `dA = -G` |

当前不做广播，输入输出 shape 相同，反向不需要归约。`G` 是上游梯度，由引擎传入。

## 第一阶段限制

- 只支持标量输出 `backward()`；非标量后续加 `backward(seed)`。
- 第一阶段只实现浮点梯度；`requires_grad_` 暂不限制 dtype，具体支持范围和检查位置待实现反向时确定。
- 视图、`clone()` 和跨设备 `to()` 尚无反向规则。当前在建图开启且 Tensor 参与 autograd（`requires_grad()` 或 `grad_node()` 非空）时抛 `std::logic_error`，避免静默断图；`detach()` 和同设备 `to()` 不受影响。
- 后续实现 `CloneNode`（恒等反向）、`CopyNode`（梯度复制回源设备）、`ViewNode`（梯度散射回基张量）后移除上述保护，使这些操作可微分。
- 先验证 CPU；Node 和引擎与设备无关，CUDA 复用同一批算子，但需要目标 GPU 环境验证。
- `zero_` / `fill_` / `copy_` 不建图；优化器更新需在 `NogradNode` 下进行。
- `detach()` 共享存储但使用新的空 meta，其梯度不会回传到原 Tensor。

## 任务清单

分工：用户实现核心代码；助手负责测试、运行验证和审查。完成一项并通过测试后再进入下一项。

当前已完成：`autograd.h` 接口、`autograd.cpp` 开关/守卫、Tensor 的 `AutogradMetadata` 初始化及全部 autograd 方法、`add/sub/mul/div/neg` 的反向 Node、`clone`/跨设备 `to`/视图的临时保护、`autograd_cpu_test` 基础用例。`RunBackward` 已写参数检查和 `NogradNode` 骨架，`TopoSort` 与梯度传递待实现；方案见上节。

### 第一天：Tensor 梯度状态与反向规则

- [x] 1. `autograd/autograd.h`：`AutogradMetadata`、`AutogradNode`、`EnableGrad`、`NogradNode`、`RunBackward` 声明。
- [x] 2. `autograd/autograd.cpp`：梯度开关和 guard；`RunBackward` 先留抛异常的桩；把该文件加入 `CMakeLists.txt` 的库源文件。
- [x] 3. `tensor/tensor.h/.cpp`：`autograd_metadata_` 成员；构造函数、拷贝/移动和 `reshape` 的 meta 初始化；`requires_grad_` / `has_grad` / `grad`。
- [x] 4. `tensor/tensor.cpp`：`grad_node()` / `set_grad_node()` / `accumulate_grad()` / `detach()` / `backward()` 及视图、`clone()`、跨设备 `to()` 的临时保护。
- [x] 5. `AddNode` + `add()` 建图；直接调用 `grad_node()->backward(seed)` 验证。
- [x] 6. `SubNode`、`NegNode`。
- [x] 7. `MulNode`、`DivNode`（保存前向输入）。
- [x] 8. 注册 `autograd_cpu_test`；当前 5/5 通过，覆盖开关、元数据、各算子直接反向和保护路径。

### 第二天：反向引擎

- [ ] 9. `RunBackward` 入口：参数检查、`NogradNode`、seed。
- [ ] 10. `TopoSort`：DFS 后序 + `visited` 去重，记录 `{node, 输出 Tensor}`；`order` 为前向拓扑序。
- [ ] 11. 逆序处理：临时梯度表以 `const AutogradNode*` 为键；中间量在自身节点处理时写持久梯度，叶子收到梯度时直接累加。
- [ ] 12. 端到端用例：链式中间 Tensor 不保存梯度但继续传递、菱形、重复 `backward()`、`detach()`、`no_grad`、错误输入。
- [ ] 13. 更新本 README、`tests/README.md` 和 `AGENTS.md` 状态。

### 后续：可微分 Tensor 操作

- [ ] 实现 `CloneNode`，反向为恒等，移除 `clone()` 保护。
- [ ] 实现 `CopyNode`，把梯度复制回源设备，移除跨设备 `to()` 保护。
- [ ] 实现 `ViewNode`，覆盖 `operator[]`、`narrow`、`reshape`，处理 stride 和重叠视图的梯度累加，移除视图保护。
- [ ] 补对应端到端测试。

测试由助手编写并运行；CUDA 反向需在有 GPU 的环境补跑。
