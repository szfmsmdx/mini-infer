# ops 设计与待办

`src/ops/` 负责 Tensor 操作的成员函数实现、数值算子及 CPU/CUDA kernel。`src/tensor/` 负责存储所有权、元数据、基础查询和打印。

已拆分公共入口、CPU 后端和 CUDA 后端。`elementwise/elementwise.cpp` 负责输入检查、输出分配与设备分流；`add()` / `sub()` / `mul()` / `div()` / `neg()` 的 CPU 和 CUDA 路径均支持 stride，并由 Tensor 运算符复用。

其他算子将在开始实现时按需建立文件和接口。

`mtensor::sqrt(x)` 逐元素开平方；`mtensor::rsqrt(x)` 直接计算平方根的倒数，不分配中间 Tensor。CPU 遍历一次输入，CUDA 调用一个使用 `rsqrtf` 的 kernel。包含 `mini_infer/ops.h` 即可使用。两者都保持 shape 和 device，支持跨步视图。整数输入先转 FP32 再计算，不建反向图；FP32、FP16 和 BF16 输出保持输入 dtype，以 FP32 计算后写回。

`silu(x)` 在 `elementwise/elementwise.cpp` 分发到 CPU/CUDA 的 `elementwise` 文件，逐元素计算完整公式，不再分配 sigmoid 中间 Tensor。通常使用 FP32 计算；`x < -80` 的负尾部用 double 计算 `x * exp(x) / (1 + exp(x))`，避免 exp 在乘 x 前下溢，最后写回输出 dtype。浮点保持 dtype，整数输入提升到 FP32。`SiluNode` 在 `elementwise/elementwise.cpp`，CUDA 反向在 `elementwise/cuda.cu` 中逐元素执行反向，采用 Torch 的 FP32 运算顺序，最后写回原 dtype；不使用 double，输入版本检查不变。CPU 和 CUDA 的双参考精度测试均通过，结果见 [测试说明](../../tests/README.md#数值精度检查)。

`add()` / `sub()` / `mul()` / `div()` 支持自动广播：右对齐推导公共 shape，在设备分发前按需创建 `broadcast()` 视图，复用 stride kernel；反向节点保存广播后的输入，由 `BroadcastNode` 沿广播维度求和。相同 shape 不创建额外视图。输入仍须同 dtype、同 device。

## 文件结构

```text
include/mini_infer/ops/
    activation.h
    elementwise.h
    matmul.h
    norm.h
    reduction.h

src/ops/
    elementwise/    # 逐元素与激活算子
        elementwise.cpp  # 前向入口、反向节点和原有 CPU 反向函数
        kernels.h   # CPU/CUDA 入口声明
        cpu.cpp     # CPU 前向
        cuda.cu     # CUDA 前向和反向
    matmul/         # matmul.cpp、kernels.h、cpu.cpp、cuda.cu
    reduction/      # reduction.cpp、nodes.h、kernels.h、cpu.cpp、cuda.cu
    norm/           # norm.cpp、kernels.h、cuda.cu
    copy/           # copy.cpp、nodes.h、kernels.h、cpu.cpp、cuda.cu
    fill/           # fill.cpp、kernels.h、cpu.cpp、cuda.cu
    layout/         # layout.cpp
    common/
        cuda_index.cuh

src/device/
    memory.h
    memory.cpp
```

本轮按现有函数整体迁移。RMSNorm 原有 CPU 循环仍属于原函数体，位于 `norm/norm.cpp`。梯度开关与通用反向引擎保留在 `src/autograd/autograd.cpp`。


按操作类别区分公共逻辑，按设备区分 kernel。后端头文件是项目内部接口，指针由 Tensor 持有。设备内存分配和释放位于 `src/device/memory.*`。

`sum()` 对全部元素归约：整数提升为 Int64 并按 64 位回绕；Float32、Float16、BFloat16 使用 Float32 累加并保持输入 dtype，支持非连续 stride 和浮点反向。
## 接口与结果所有权

当前公共步骤按使用范围组织。

| 公共步骤 | 位置 | 调用范围 |
| --- | --- | --- |
| `ContiguousStride()` | `layout/layout.cpp` | Tensor 构造、reshape、clone 和设备迁移 |
| `PrepareBinaryInputs()` | `elementwise/elementwise.cpp` | add、sub、mul、div 的检查和广播准备 |
| `Tensor::MakeView()` | `layout/layout.cpp` | 8 种视图操作的共享存储、独立梯度元数据和 CUDA 布局初始化 |
| `AllocateStorage()` | `src/device/memory.*` | 构造、clone 和跨设备迁移的数据分配与释放器绑定 |
| `ForEachOffset()` | `elementwise/cpu_iteration.h` | CPU 逐元素前向和反向的 stride 遍历 |
| `LaunchUnary()`、`LaunchBackward()` | `elementwise/cuda.cu` | 4 个一元前向和 3 个反向的 CUDA 启动准备 |

视图的形状推导、参数检查和 Node 状态保存仍由对应操作完成。CPU 遍历回调负责元素计算，偏移单位为元素；连续输入的 rsqrt 快路径直接循环。CUDA 启动函数使用编译期 Kernel 参数，共用线程块和布局参数准备。

以加法为例，计划同时提供：

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

`add()` 返回拥有独立存储的新 Tensor，不修改输入，也不与输入共享输出存储。返回 Tensor 按现有的移动语义或返回值消除传递，不需要为返回值额外复制整块数据。

`mul()` / `operator*` 表示对应位置逐元素相乘；矩阵乘法使用独立的 `matmul()` 接口。

两个 Tensor 的 dtype 和 device 必须相同，shape 按广播规则匹配。`x + eps`、`2 - x`、`x * 0.5f`、`1 / x` 等运算支持标量在任一侧。标量按 `fill_` 的规则转换为与 x 同 dtype、同 device 的零维 Tensor，再广播计算。因此整数 Tensor 与浮点标量混算时，先把标量转为整数，不会提升整个 Tensor。

从一开始按多 dtype 组织计算：普通数值类型使用模板，FP16/BF16 使用专门的数值转换或 CUDA 浮点类型。实现尚未覆盖的 dtype/device 必须明确报错，不能统一按 `float*` 解释。

CPU 和 CUDA 逐元素运算支持不同 stride 的非连续输入；CUDA 输出为连续存储。零维标量按单个元素计算，默认空对象单独拒绝。

## 反向传播（WIP）

前向计算完成后，如果建图开关开启且 dtype 支持反向，算子创建一个 Node 并挂到输出上，与输入的 `requires_grad` 无关。逐元素 Node 位于 `src/ops/elementwise/elementwise.cpp`。推理时用守卫关闭建图。

| 算子 | Node | 输入梯度 |
| --- | --- | --- |
| `add(a, b)` | `AddNode` | `dA = G`, `dB = G` |
| `sub(a, b)` | `SubNode` | `dA = G`, `dB = -G` |
| `mul(a, b)` | `MulNode` | `dA = G * b`, `dB = G * a` |
| `div(a, b)` | `DivNode` | `dA = G / b`, `dB = -G * a / (b * b)` |
| `neg(a)` | `NegNode` | `dA = -G` |
| `sqrt(x)` | `SqrtNode` | `dX = G / (2 * sqrt(x))` |
| `exp(x)` | `ExpNode` | `dX = G * y`，复用前向输出 `y = exp(x)`，按所在设备执行逐元素乘法 |
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

`elementwise/elementwise.cpp` 统一检查参数、分配输出，再根据 device 调用普通后端入口：CPU 为 `AddCPU()` / `SubCPU()` / `MulCPU()` / `DivCPU()`，CUDA 为对应的 `AddCUDA()` / `SubCUDA()` / `MulCUDA()` / `DivCUDA()`。该文件不包含元素循环或 dtype 分发。CUDA 调用放在 `#ifdef ENABLE_CUDA` 中；CPU 构建不能引用未链接的 CUDA 启动函数。

CUDA 启动函数按 dtype 分发 kernel，读取设备上的 shape 和各输入的 stride。

`mini_infer/ops/elementwise.h` 包含 `mini_infer/tensor.h`。Tensor 运算符在头文件里声明，在 `src/ops/elementwise/elementwise.cpp` 中转发给对应算子。

### 按 dtype 分发

对外的 `add(const Tensor&, const Tensor&)` 保持普通函数，Tensor 的 dtype 在运行时决定。CPU 入口也使用普通函数，由它在 CPU 文件内选择模板实例：

```cpp
// backend.h：内部后端入口。
void AddCPU(const void* a, const void* b, void* out, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& out_stride);

// src/ops/elementwise/cpu.cpp：匿名命名空间内的计算模板。
template <typename T>
void AddTypedCPU(const T* a, const T* b, T* out, const std::vector<int>& shape,
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

`add()` / `sub()` / `mul()` 的整数溢出采用固定位宽回绕：先用对应的无符号类型计算，再把位模式写回原类型；边界值已由 CPU 测试覆盖。整数 `div()` 向零截断，除零返回 0，最小值除以 -1 回绕为最小值。这些是项目自定义的规则，CPU 和后续 CUDA 实现必须保持一致。浮点四则运算遵循 IEEE 754，溢出得到无穷，NaN 继续传播。各 dtype 使用与其精度相符的误差范围。

## TODO

按以下顺序推进，每一步先明确接口，再写实现和对应验证。

- [x] 声明逐元素算子、CUDA 启动函数、Tensor 运算符、分配构造函数和只读数据访问接口。
- [x] 补齐只按 shape、dtype、device 分配存储的 Tensor 接口；只分配不初始化，避免为了分配输出先准备一份 CPU 源数组。
- [x] 实现 `const void* data_ptr() const`，使算子能读取 `const Tensor&` 输入。
- [ ] 完善公共输入检查：拒绝空对象，检查 shape、dtype、device，并拒绝尚未实现的类型或设备。
- [x] 拆分公共入口、CPU 文件与 CUDA 文件，声明可跨文件调用的普通后端入口。
- [x] 将加法计算移入 CPU 文件的 `AddTypedCPU<T>()`，按 stride 遍历元素。
- [x] 完成 `add()` / `sub()` / `mul()` / `div()` 的 CPU 和 CUDA 路径：检查输入、一次分配独立输出、按 device/dtype 分发并调用计算函数。
- [x] 在 `src/ops/elementwise/cpu.cpp` 的 CPU 入口内使用 `switch`，选择对应的内部模板实例。
- [x] 将 FP16/BF16 解码函数整理到 `dtype.h`，供算子复用。
- [x] 实现 CPU 的 FP16/BF16 解码、FP32 计算和编码路径。
- [x] `add()` / `sub()` / `mul()` 的整数溢出采用固定位宽回绕，并覆盖各整数类型的边界值。
- [x] 整数 `div()` 向零截断，除零返回 0，最小值除以 -1 回绕为最小值，并覆盖边界测试。
- [x] 浮点四则运算遵循 IEEE 754，覆盖无穷和 NaN 传播。
- [x] 声明并实现 `Tensor::operator+` / `operator-` / `operator*` / `operator/` 以及一元 `operator-`。
- [x] 完成 CPU 的 `neg()`：整数取负按固定位宽回绕，FP16/BF16 翻转符号位。
- [x] 将 `elementwise/elementwise.cpp`、`src/ops/elementwise/cpu.cpp` 和 CPU 测试接入 CMake，覆盖各 dtype、标量、多维输入、索引视图、输入不变、输出独立和无效输入。
- [x] 实现支持输入 stride 的 `AddCUDA()` / `SubCUDA()` / `MulCUDA()` / `DivCUDA()` / `NegCUDA()`；处理 dtype、线程越界和启动错误。
- [x] 按 `ENABLE_CUDA` 条件接入 CUDA 测试；对照 CPU 结果，覆盖元素数不整除线程块大小的情况，读取结果时检查执行错误。
- [x] 二元算子支持广播，四则运算符支持 Tensor 与标量混算。
- [ ] 设计 `add_()` / `operator+=` 等原地接口。
- [x] `AddNode` / `SubNode` / `NegNode` 反向规则。
- [x] `MulNode` / `DivNode` 保存前向输入，并在 `NogradNode` 下复用现有算子计算梯度。
- [x] 按建图开关和支持反向的 dtype 建图；设计见 [autograd/README.md](../autograd/README.md)。

### 扩展算子

- [ ] 按需建立扩展算子的接口和实现。
- [ ] 新算子实现时同步补 Node 和反向公式。
- [x] 实现二维 `matmul()` 的 CPU/CUDA 前向，支持 8 种 dtype 和非连续输入。Colab CPU/CUDA 回归 15/15 通过。
- [x] 二维浮点 `matmul()` 接入反向：`grad_A = G @ Bᵀ`，`grad_B = Aᵀ @ G`，其中 G 为输出梯度。
- [ ] 实现 `linear()`。
- [ ] 实现 `rms_norm()` / `rms_norm_zero_centered()` / `rms_norm_gated()` / `layer_norm()`。
- [x] 实现 `exp()` 的 CPU/CUDA 前向与基于输出的反向。
- [x] `sigmoid()` 和 `silu()` 对整数输入返回 FP32，对浮点输入保持 dtype；SiLU 使用独立逐元素 CPU/CUDA 前向，避免 sigmoid 中间 Tensor 提前舍入。
- [ ] 实现 `softplus()`。
- [x] 实现全元素 `sum()` 的 CPU/CUDA 前向与反向，并在 Colab A100 验证 CUDA 测试。
- [ ] 实现指定维度的 `sum()`，以及 `mean()` / `max()` / `softmax()` / `argmax()` / `topk()`。
- [ ] 实现 `embedding()` / `index_select()` / `gather()` / `scatter_add()`。
- [ ] 实现 `rope()`；确定 MRoPE 接口。
- [ ] 实现 `causal_attention()`。
- [ ] 实现 `depthwise_causal_conv1d()`。
- [ ] 实现 `gated_delta_rule()`。

测试与构建方式见 [tests/README.md](../../tests/README.md)。
