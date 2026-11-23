# 算子设计与待办

算子入口检查输入、分配输出、选择 CPU 或 CUDA，并在启用建图时记录反向节点。后端按 dtype 和 stride 读取 Tensor 数据。

## 文件结构

| 目录 | 内容 |
| --- | --- |
| `elementwise/` | 四则运算、开方、指数、对数 |
| `activation/` | Sigmoid、SiLU、Softmax |
| `matmul/`、`reduction/`、`norm/` | 矩阵乘、归约、RMSNorm |
| `attention/`、`embedding/`、`loss_fn/`、`position_embedding/` | 因果注意力、词嵌入、交叉熵、RoPE |
| `copy/`、`fill/`、`layout/` | 存储复制、设备和 dtype 转换、填充与视图 |
| `common/` | 输入校验、CPU/CUDA stride 索引 |

普通算子按类别放置入口和反向节点。逐元素、矩阵乘等算子使用同目录的 `cpu.cpp`、`cuda.cu`；因果注意力、词嵌入、交叉熵和 RoPE 各用同名的 `.cpp`、`.cu`。

RoPE 的公开声明在 [`mini_infer/ops/position_embedding.h`](../../include/mini_infer/ops/position_embedding.h)。

## 接口与结果所有权

| 公共步骤 | 位置 | 调用范围 |
| --- | --- | --- |
| `ContiguousStride()` | `layout/layout.cpp` | Tensor 构造、reshape、clone 和设备迁移 |
| `PrepareBinaryInputs()` | `elementwise/elementwise.cpp` | add、sub、mul、div 的检查和广播准备 |
| `Tensor::MakeView()` | `layout/layout.cpp` | 8 种视图操作的共享存储、独立梯度元数据和 CUDA 布局初始化 |
| `AllocateStorage()` | `src/device/memory.*` | 构造、clone 和跨设备迁移的数据分配与释放器绑定 |
| `ForEachOffset()` | `common/cpu_iteration.h` | CPU 逐元素算子和 dtype 转换的 stride 遍历 |
| `LaunchUnary()`、`LaunchBackward()` | `common/cuda_unary.cuh` | 一元算子的 CUDA 启动参数和 stride 准备 |

视图的形状推导、参数检查和 Node 状态保存仍由对应操作完成。CPU 遍历回调负责元素计算，偏移单位为元素；连续输入的 rsqrt 快路径直接循环。CUDA 启动函数使用编译期 Kernel 参数，共用线程块和布局参数准备。

逐元素运算提供成员运算符和普通函数。

```cpp
// Tensor 类内，位于 namespace mtensor。
Tensor operator+(const Tensor& other) const;

// 普通函数，位于 namespace mtensor。
Tensor add(const Tensor& a, const Tensor& b);

// 一元负号及其普通函数形式。
Tensor operator-() const;
Tensor neg(const Tensor& a);
```

`operator+` 只调用 `add(*this, other)`，参数检查和计算逻辑统一写在 `add()` 中。减、乘、除和一元负号按相同方式组织。

`add()` 将结果写入新分配的 Tensor，通过移动或返回值消除返回。

`mul()` / `operator*` 表示对应位置逐元素相乘；矩阵乘法使用独立的 `matmul()` 接口。

两个 Tensor 的 dtype 和 device 必须相同，shape 按广播规则匹配。`x + eps`、`2 - x`、`x * 0.5f`、`1 / x` 等运算支持标量在任一侧。标量按 `fill_` 的规则转换为与 x 同 dtype、同 device 的零维 Tensor，再广播计算。因此整数 Tensor 与浮点标量混算时，先把标量转为整数，不会提升整个 Tensor。

普通数值类型使用模板计算。FP16/BF16 使用对应的数值转换或 CUDA 浮点类型。算子遇到未支持的 dtype 或 device 时抛出异常。

CPU 和 CUDA 逐元素运算支持不同 stride 的非连续输入；CUDA 输出为连续存储。零维标量按单个元素计算，默认空对象单独拒绝。

## 反向传播

前向计算完成后，如果建图开关开启且 dtype 支持反向，算子创建一个 Node 并挂到输出上，与输入的 `requires_grad` 无关。Node 和算子入口放在一起：Sigmoid、SiLU 在 `activation/activation.cpp`，Softmax 在 `activation/softmax.cpp`。推理时用守卫关闭建图。

| 算子 | Node | 输入梯度 |
| --- | --- | --- |
| `add(a, b)` | `AddNode` | `dA = G`, `dB = G` |
| `sub(a, b)` | `SubNode` | `dA = G`, `dB = -G` |
| `mul(a, b)` | `MulNode` | `dA = G * b`, `dB = G * a` |
| `div(a, b)` | `DivNode` | `dA = G / b`, `dB = -G * a / (b * b)` |
| `neg(a)` | `NegNode` | `dA = -G` |
| `sqrt(x)` | `SqrtNode` | `dX = G / (2 * sqrt(x))` |
| `exp(x)` | `ExpNode` | `dX = G * y`，复用前向输出 `y = exp(x)`，按所在设备执行逐元素乘法 |
| `log(x)` | `LogNode` | `dX = G / x`，使用前向输入并检查其版本 |
| `sigmoid(x)` | `SigmoidNode` | 令 `t = exp(-abs(x))`，在 FP32 中计算 `dX = (G * t) / ((1 + t) * (1 + t))`，最后写回原 dtype |
| `silu(x)` | `SiluNode` | FP32 计算 `s = 1 / (1 + exp(-x))`，再计算 `(G * s) * (1 + x * (1 - s))`，最后写回原 dtype |
| `rsqrt(x)` | `RsqrtNode` | 正数输入：令 `y = 1 / sqrt(x)`，逐元素计算 `dX = -0.5 * G * y³` |

表中的 `G` 是引擎传入的上游梯度。反向在 `NogradNode` 下执行，不会继续建图。`rsqrt` 的反向在 `ops/elementwise/elementwise.cpp` 和 `ops/elementwise/cuda.cu` 中逐元素执行：`y` 和 `((-0.5 * y) * G) * y * y` 的中间结果保持 FP32，最后一次写回原 dtype，避免低精度中间 Tensor 反复舍入。

## 分发流程

```text
a + b
  ↓
Tensor::operator+                         src/ops/elementwise/elementwise.cpp
  ↓
add(a, b)                                src/ops/elementwise/elementwise.cpp
  ├─ 检查输入 shape、dtype、device 和空状态
  ├─ 分配与输入具有相同元数据的独立输出
  ├─ CPU：调用 AddCPU(...)                src/ops/elementwise/cpu.cpp
  │    ├─ 根据 dtype 分发
  │    ├─ 普通数值类型：AddTypedCPU<T>(...)，按 stride 遍历
  │    └─ FP16/BF16：解码为 float → 计算 → 编码并写回
  └─ CUDA：调用 AddCUDA(...)              src/ops/elementwise/cuda.cu
             按各输入 stride 计算偏移并写入连续输出
```

### 按 device 分发

`elementwise/elementwise.cpp` 检查参数、分配输出，再按 device 调用后端。`AddCPU()` / `SubCPU()` / `MulCPU()` / `DivCPU()` 是 CPU 入口，`AddCUDA()` / `SubCUDA()` / `MulCUDA()` / `DivCUDA()` 在启用 CUDA 的构建中调用。

CUDA 启动函数按 dtype 分发 kernel，读取设备上的 shape 和各输入的 stride。

`mini_infer/ops/elementwise.h` 包含 `mini_infer/tensor.h`。Tensor 运算符在头文件里声明，在 `src/ops/elementwise/elementwise.cpp` 中转发给对应算子。

### 按 dtype 分发

对外的 `add(const Tensor&, const Tensor&)` 保持普通函数，Tensor 的 dtype 在运行时决定。CPU 入口也使用普通函数，由它在 CPU 文件内选择模板实例：

```cpp
// kernels.h：内部后端入口。
void AddCPU(const void* a, const void* b, void* out, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& out_stride);

// src/ops/elementwise/cpu.cpp：匿名命名空间内的计算模板。
template <typename T>
void AddTypedCPU(const void* a, const void* b, void* out, const std::vector<int>& shape,
                 const std::vector<std::size_t>& a_stride, const std::vector<std::size_t>& b_stride,
                 const std::vector<std::size_t>& out_stride);
```

`AddTypedCPU<T>()` 的定义放在 `src/ops/elementwise/cpu.cpp` 的匿名命名空间内，由同文件的 `AddCPU()` 实例化调用。跨文件只调用普通入口，避免模板定义不可见导致链接错误。输入指针只读、输出指针可写，均不转移所有权；模板按 shape 和 stride 遍历并计算，不负责分配、形状检查或设备选择。减、乘、除使用对应的内部模板函数。

- CPU 路径在 `src/ops/elementwise/cpu.cpp` 中用 `switch` 选择普通数值类型，再调用 `AddTypedCPU<float>()`、`AddTypedCPU<std::int32_t>()` 等实例。
- CUDA 启动入口按 dtype 选择普通数值类型或半精度 kernel。
- `void*` 只描述地址；必须根据 dtype 转换成正确的元素指针，输入使用只读指针，输出使用可写指针。

FP16/BF16 使用 16 位编码，不能直接将编码作为整数参与浮点运算；它们在同一个 `switch` 中分发到专门的半精度计算路径：

- CPU：四则运算按 stride 遍历，将半精度编码转换为 `float`、计算后重新编码；取负共用符号位处理。
- CUDA：解码为 float 计算，再编码回 FP16/BF16；取负翻转符号位。

`add()` / `sub()` / `mul()` 的整数溢出采用固定位宽回绕：先用对应的无符号类型计算，再把位模式写回原类型；边界值已由 CPU 测试覆盖。整数 `div()` 向零截断，除零返回 0，最小值除以 -1 回绕为最小值。CPU 与 CUDA 共用这些整数规则。浮点四则运算遵循 IEEE 754，溢出得到无穷，NaN 继续传播。各 dtype 使用与其精度相符的误差范围。

## 已实现的计算

- `sqrt()`、`rsqrt()`、`exp()`、`log()`、`sigmoid()`、`silu()` 支持 stride。整数输入提升到 FP32。
- `softmax(x, dim)` 支持 CPU/CUDA 的 8 种 dtype，默认沿最后一维计算。减最大值、指数、求和和除法都在 FP32 中完成；浮点输出保留 dtype，整数先转 FP32 后计算、不建反向节点。低精度输入的反向用未舍入的 FP32 概率计算梯度，最后转回输入 dtype。
- `sum()` 支持全元素和指定维度归约。浮点用 FP32 累加，结果保持输入 dtype。整数结果为 Int64，按 64 位回绕。`mean()` 支持全元素和指定维度。
- `max()`、`min()` 支持 CPU/CUDA 的全元素和指定维度归约，沿用负维度和 `keepdim`。支持非连续输入，只返回极值，保留输入 dtype 和 device；浮点输入的并列极值平分上游梯度。归约组含 NaN 时返回 NaN，反向只在该组的 NaN 之间平分梯度。
- 二维 `matmul()` 的 CPU/CUDA 前向支持 8 种 dtype。浮点反向计算 `dA = G @ Bᵀ` 和 `dB = Aᵀ @ G`。
- FP16/BF16 RMSNorm 在 CPU 上用 `to(Dtype::kFloat32)` 和 Tensor 操作，在 CUDA 上用专用 kernel。两条路径都在 FP32 中计算，最后才写回低精度输出和梯度。SiLU 前向用 double 处理 `x < -80` 的负尾，反向使用 FP32。[精度排查](../../note/code/算子精度问题.md)
- 词嵌入、RoPE、因果注意力和交叉熵均接入 CPU/CUDA 前向及反向。[GPT 示例](../../examples/nanogpt/README.md)
- CPU 因果注意力用 `narrow` 按头和可见前缀取视图，前向复用 `matmul` 和 `softmax`，反向用 Tensor 运算累加 K/V 梯度；CUDA 使用专用 kernel。

## 待办

- [ ] 原地算子接口，例如 `add_()` 和 `operator+=`。
- [ ] `softplus()`、`argmax()` 和 `topk()`。
- [ ] `index_select()`、`gather()` 和 `scatter_add()`。
- [ ] `rms_norm_zero_centered()`、`rms_norm_gated()` 和 `layer_norm()`。
- [ ] MRoPE、`depthwise_causal_conv1d()` 和 `gated_delta_rule()`。

[构建与测试](../../tests/README.md)
