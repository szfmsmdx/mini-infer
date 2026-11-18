# ops 设计与待办

`ops/` 负责 Tensor 操作的成员函数实现、数值算子及 CPU/CUDA kernel；`tensor/` 负责存储所有权、元数据和基础查询。

已拆分公共入口、CPU 后端和 CUDA 后端。`elementwise.cpp` 负责输入检查、输出分配与设备分流；`add()` / `sub()` / `mul()` / `div()` / `neg()` 的 CPU 和 CUDA 路径均支持 stride，并由 Tensor 运算符复用。

其他算子将在开始实现时按需建立文件和接口。

`add()` / `sub()` / `mul()` / `div()` 支持自动广播：右对齐推导公共 shape，在设备分发前按需创建 `broadcast()` 视图，复用 stride kernel；反向节点保存广播后的输入，由 `BroadcastNode` 沿广播维度求和。相同 shape 不创建额外视图。输入仍须同 dtype、同 device。

## 文件结构

```text
ops/
    elementwise.h           # 数值算子接口
    elementwise.cpp         # 数值算子检查、设备分流与建图
    reduction.h             # 归约算子接口
    reduction.cpp           # 归约算子入口、设备检查与建图
    tensor_ops.cpp          # Tensor 操作的成员函数定义
    backend.h               # CPU / CUDA 后端函数的内部声明
    cpu/
        elementwise.cpp     # CPU 逐元素计算
        reduction.cpp       # CPU 全元素归约
        data_ops.cpp        # CPU 复制、填充、清零、打印格式化
    cuda/
        elementwise.cu      # CUDA 逐元素计算
        data_ops.cu         # CUDA 复制、填充、清零及设备间数据传输
        index.cuh           # CUDA 逻辑下标到 stride 偏移的辅助函数
        reduction.cu        # CUDA 全元素归约
    README.md
```

按操作类别区分公共逻辑，按设备区分 kernel。后端头文件是项目内部接口，指针由 Tensor 持有，kernel 不接管所有权。当前设备内存管理仍位于 `utils/memory.*`，按需引入设备资源层。

`sum()` 对全部元素归约：整数提升为 Int64 并按 64 位回绕；Float32、Float16、BFloat16 使用 Float32 累加并保持输入 dtype，支持非连续 stride 和浮点反向。
## 接口与结果所有权

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

要求两个输入非空，且 shape、dtype、device 相同。输出保持相同的 shape、dtype、device。不做广播、隐式类型转换、跨设备计算或原地版本。

从一开始按多 dtype 组织计算：普通数值类型使用模板，FP16/BF16 使用专门的数值转换或 CUDA 浮点类型。实现尚未覆盖的 dtype/device 必须明确报错，不能统一按 `float*` 解释。

CPU 和 CUDA 逐元素运算支持不同 stride 的非连续输入；CUDA 输出为连续存储。零维标量按单个元素计算，默认空对象单独拒绝。

## 反向传播（WIP）

前向计算完成后，如果建图开关开启且 dtype 支持反向，算子创建一个 Node 并挂到输出上，与输入的 `requires_grad` 无关。Node 类放在 `autograd/backward_nodes/elementwise.h/.cpp`，公共接口不变；推理时用守卫关闭建图。

| 算子 | Node | 输入梯度 |
| --- | --- | --- |
| `add(a, b)` | `AddNode` | `dA = G`, `dB = G` |
| `sub(a, b)` | `SubNode` | `dA = G`, `dB = -G` |
| `mul(a, b)` | `MulNode` | `dA = G * b`, `dB = G * a` |
| `div(a, b)` | `DivNode` | `dA = G / b`, `dB = -G * a / (b * b)` |
| `neg(a)` | `NegNode` | `dA = -G` |

所有 Node 都通过 `inputs` 保存前向输入以连接计算图；`MulNode` 和 `DivNode` 的反向公式还需读取输入数值。反向公式复用现有 CPU/CUDA 算子，在 `NogradNode` 下执行，不会继续建图。当前没有广播，输入输出 shape 相同，反向不需要归约。`G` 是上游梯度，由 autograd 引擎传入。

## 分发流程

```text
a + b
  ↓
Tensor::operator+                         ops/tensor_ops.cpp
  ↓
add(a, b)                                ops/elementwise.cpp
  ├─ 检查输入 shape、dtype、device 和空状态
  ├─ 分配与输入具有相同元数据的独立输出
  ├─ CPU：调用 AddCPU(...)                ops/cpu/elementwise.cpp
  │    ├─ 根据 dtype 分发
  │    ├─ 普通数值类型：AddTypedCPU<T>(...)，按 stride 遍历
  │    └─ FP16/BF16：解码为 float → 计算 → 编码并写回
  └─ CUDA：调用 AddCUDA(...)      ops/cuda/elementwise.cu
             按各输入 stride 计算偏移并写入连续输出
```

### 按 device 分发

`elementwise.cpp` 统一检查参数、分配输出，再根据 device 调用普通后端入口：CPU 为 `AddCPU()` / `SubCPU()` / `MulCPU()` / `DivCPU()`，CUDA 为对应的 `AddCUDA()` / `SubCUDA()` / `MulCUDA()` / `DivCUDA()`。该文件不包含元素循环或 dtype 分发。CUDA 调用放在 `#ifdef ENABLE_CUDA` 中；CPU 构建不能引用未链接的 CUDA 启动函数。

CUDA 启动函数按 dtype 分发 kernel，读取设备上的 shape 和各输入的 stride。

`elementwise.h` 包含 `tensor/tensor.h`；`tensor_ops.cpp` 包含 `ops/elementwise.h`。`tensor.h` 只声明运算符，不反过来包含算子头文件。

### 按 dtype 分发

对外的 `add(const Tensor&, const Tensor&)` 保持普通函数，Tensor 的 dtype 在运行时决定。CPU 入口也使用普通函数，由它在 CPU 文件内选择模板实例：

```cpp
// backend.h：内部后端入口。
void AddCPU(const void* a, const void* b, void* out, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& out_stride);

// ops/cpu/elementwise.cpp：匿名命名空间内的计算模板。
template <typename T>
void AddTypedCPU(const T* a, const T* b, T* out, const std::vector<int>& shape,
                 const std::vector<std::size_t>& a_stride, const std::vector<std::size_t>& b_stride,
                 const std::vector<std::size_t>& out_stride);
```

`AddTypedCPU<T>()` 的定义放在 `ops/cpu/elementwise.cpp` 的匿名命名空间内，由同文件的 `AddCPU()` 实例化调用。跨文件只调用普通入口，避免模板定义不可见导致链接错误。输入指针只读、输出指针可写，均不转移所有权；模板按 shape 和 stride 遍历并计算，不负责分配、形状检查或设备选择。减、乘、除使用对应的内部模板函数。

- CPU 路径在 `ops/cpu/elementwise.cpp` 中用 `switch` 选择普通数值类型，再调用 `AddTypedCPU<float>()`、`AddTypedCPU<std::int32_t>()` 等实例。
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
- [x] 在 `ops/cpu/elementwise.cpp` 的 CPU 入口内使用 `switch`，选择对应的内部模板实例。
- [x] 将 FP16/BF16 解码函数整理到 `dtype.h`，供算子复用。
- [x] 实现 CPU 的 FP16/BF16 解码、FP32 计算和编码路径。
- [x] `add()` / `sub()` / `mul()` 的整数溢出采用固定位宽回绕，并覆盖各整数类型的边界值。
- [x] 整数 `div()` 向零截断，除零返回 0，最小值除以 -1 回绕为最小值，并覆盖边界测试。
- [x] 浮点四则运算遵循 IEEE 754，覆盖无穷和 NaN 传播。
- [x] 声明并实现 `Tensor::operator+` / `operator-` / `operator*` / `operator/` 以及一元 `operator-`。
- [x] 完成 CPU 的 `neg()`：整数取负按固定位宽回绕，FP16/BF16 翻转符号位。
- [x] 将 `elementwise.cpp`、`ops/cpu/elementwise.cpp` 和 CPU 测试接入 CMake，覆盖各 dtype、标量、多维输入、索引视图、输入不变、输出独立和无效输入。
- [x] 实现支持输入 stride 的 `AddCUDA()` / `SubCUDA()` / `MulCUDA()` / `DivCUDA()` / `NegCUDA()`；处理 dtype、线程越界和启动错误。
- [x] 按 `ENABLE_CUDA` 条件接入 CUDA 测试；对照 CPU 结果，覆盖元素数不整除线程块大小的情况，读取结果时检查执行错误。
- [ ] 后续独立设计广播、标量混合运算及 `add_()` / `operator+=` 等原地接口。
- [x] `AddNode` / `SubNode` / `NegNode` 反向规则。
- [x] `MulNode` / `DivNode` 保存前向输入，并在 `NogradNode` 下复用现有算子计算梯度。
- [x] 按建图开关和支持反向的 dtype 建图；设计见 [autograd/README.md](../autograd/README.md)。

### 扩展算子

- [ ] 按需建立扩展算子的接口和实现。
- [ ] 新算子实现时同步补 Node 和反向公式。
- [x] 实现二维 `matmul()` 的 CPU/CUDA 前向，支持 8 种 dtype 和非连续输入。Colab CPU/CUDA 回归 15/15 通过。
- [ ] 实现 `matmul()` 反向及 `linear()`。
- [ ] 实现 `rms_norm()` / `rms_norm_zero_centered()` / `rms_norm_gated()` / `layer_norm()`。
- [ ] 实现 `silu()` / `sigmoid()` / `softplus()` / `exp()`。
- [x] 实现全元素 `sum()` 的 CPU/CUDA 前向与反向，并在 Colab A100 验证 CUDA 测试。
- [ ] 实现指定维度的 `sum()`，以及 `mean()` / `max()` / `softmax()` / `argmax()` / `topk()`。
- [ ] 实现 `embedding()` / `index_select()` / `gather()` / `scatter_add()`。
- [ ] 实现 `rope()`；确定 MRoPE 接口。
- [ ] 实现 `causal_attention()`。
- [ ] 实现 `depthwise_causal_conv1d()`。
- [ ] 实现 `gated_delta_rule()`。

测试与构建方式见 [tests/README.md](../tests/README.md)。
