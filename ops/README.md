# ops 设计与待办

`ops/` 负责 Tensor 的数值计算；`tensor/` 继续负责存储、形状、dtype、device、共享视图和数据复制。

已拆分公共入口、CPU 后端和 CUDA 后端。`elementwise.cpp` 保留输入检查与设备分流；原加法循环已移入 `elementwise_cpu.cpp` 的内部模板。CPU dtype 分发及 CUDA 计算仍为 TODO，占位入口会抛出未实现异常；当前尚不能执行完整加法，算子也尚未接入 CMake。

## 文件结构

```text
ops/
    README.md
    elementwise.h        # 公共算子、CPU 入口和 CUDA 启动函数的声明
    elementwise.cpp      # 参数检查、输出分配、按 device 分流
    elementwise_cpu.cpp  # CPU 入口、dtype 分发、内部模板和 FP16/BF16 计算
    elementwise_cuda.cu  # CUDA 启动入口、dtype 分发和 GPU kernel
```

先按运算类别组织文件，加减乘除放在一起。后续矩阵乘法可使用 `matmul.h/.cpp/.cu`；归约、Softmax 等做到时再划分，不预先创建空目录或抽象层。

## 接口与结果所有权

以加法为例，计划同时提供：

```cpp
// Tensor 类内，位于 namespace mtensor。
Tensor operator+(const Tensor& other) const;

// 普通函数，位于 namespace mtensor。
Tensor add(const Tensor& a, const Tensor& b);
```

`operator+` 只调用 `add(*this, other)`，参数检查和计算逻辑统一写在 `add()` 中。减、乘、除按相同方式组织。

`add()` 返回拥有独立存储的新 Tensor，不修改输入，也不与输入共享输出存储。返回 Tensor 按现有的移动语义或返回值消除传递，不需要为返回值额外复制整块数据。

`mul()` / `operator*` 表示对应位置逐元素相乘；矩阵乘法使用独立的 `matmul()` 接口。

要求两个输入非空，且 shape、dtype、device 相同。输出保持相同的 shape、dtype、device。不做广播、隐式类型转换、跨设备计算或原地版本。

从一开始按多 dtype 组织计算：普通数值类型使用模板，FP16/BF16 使用专门的数值转换或 CUDA 浮点类型。实现尚未覆盖的 dtype/device 必须明确报错，不能统一按 `float*` 解释。

当前 Tensor 及其索引视图都是连续存储，逐元素运算直接遍历全部 `numel()`，不按维度递归。零维标量的元素数为 1，使用同一段计算循环；默认空对象单独拒绝。输出只分配一次，无需按行创建临时 Tensor、递归 clone 或复制子结果。

## 分发流程

```text
a + b
  ↓
Tensor::operator+                         tensor/tensor.cpp
  ↓
add(a, b)                                ops/elementwise.cpp
  ├─ 检查输入 shape、dtype、device 和空状态
  ├─ 分配与输入具有相同元数据的独立输出
  ├─ CPU：调用 AddCPU(...)                ops/elementwise_cpu.cpp
  │    ├─ 根据 dtype 分发
  │    ├─ 普通数值类型：AddTypedCPU<T>(...)，遍历全部 numel()
  │    └─ FP16/BF16：解码为 float → 计算 → 编码并写回
  └─ CUDA：调用 LaunchAddKernel(...)
                    ↓                    ops/elementwise_cuda.cu
             选择类型、配置线程与线程块
                    ↓
             AddKernel<<<...>>>(...)
                    ↓
             检查 kernel 启动错误
```

### 按 device 分发

`elementwise.cpp` 统一检查参数、分配输出，再根据 device 调用普通后端入口：CPU 为 `AddCPU()` / `SubCPU()` / `MulCPU()` / `DivCPU()`，CUDA 为对应的 `Launch...Kernel()`。该文件不包含元素循环或 dtype 分发。CUDA 调用放在 `#ifdef ENABLE_CUDA` 中；CPU 构建不能引用未链接的 CUDA 启动函数。

CUDA 路径调用的是普通 CPU 函数 `LaunchAddKernel()`，由该函数在 `.cu` 内使用 `<<<...>>>` 启动 GPU kernel。启动函数通过头文件声明连接两边；不直接 include `.cu` 文件。

`elementwise.h` 包含 `tensor/tensor.h`；`tensor.cpp` 包含 `ops/elementwise.h`。`tensor.h` 只声明运算符，不反过来包含算子头文件。

### 按 dtype 分发

对外的 `add(const Tensor&, const Tensor&)` 保持普通函数，Tensor 的 dtype 在运行时决定。CPU 入口也使用普通函数，由它在 CPU 文件内选择模板实例：

```cpp
// elementwise.h：可跨文件调用的普通函数。
void AddCPU(const void* a, const void* b, void* out, std::size_t count, Dtype dtype);

// elementwise_cpu.cpp：匿名命名空间内的计算模板。
template <typename T>
void AddTypedCPU(const T* a, const T* b, T* out, std::size_t count);
```

`AddTypedCPU<T>()` 的定义放在 `elementwise_cpu.cpp` 的匿名命名空间内，由同文件的 `AddCPU()` 实例化调用。跨文件只调用普通入口，避免模板定义不可见导致链接错误。输入指针只读、输出指针可写，均不转移所有权；模板只负责遍历 `count` 个元素并计算，不负责分配、形状检查或设备选择。减、乘、除使用对应的内部模板函数。

- CPU 路径在 `elementwise_cpu.cpp` 中用 `DISPATCH_DTYPE` 选择普通数值类型，再调用 `AddTypedCPU<float>()`、`AddTypedCPU<std::int32_t>()` 等实例。
- CUDA 路径在 `.cu` 的启动函数中选择元素类型，启动对应的模板 kernel。
- `void*` 只描述地址；必须根据 dtype 转换成正确的元素指针，输入使用只读指针，输出使用可写指针。

FP16/BF16 当前用 `uint16_t` 保存编码，不能用 `AddTypedCPU<uint16_t>()` 对编码做整数加法。这两种 dtype 在进入普通数值分发宏前单独处理：

- CPU：读取编码，解码成 `float`，进行 FP32 中间计算，再通过 `EncodeFloat16()` / `EncodeBFloat16()` 舍入并存回 16 位编码。`dtype.h` 已提供可复用的 `DecodeFloat16()` / `DecodeBFloat16()`。
- CUDA：使用 `<cuda_fp16.h>` 的 `__half`、`<cuda_bf16.h>` 的 `__nv_bfloat16` 和对应算术函数，实现 kernel。

整数除零、整数溢出、浮点除零及特殊值处理需要明确规则，并用对应测试验证。各 dtype 使用与其精度相符的误差范围。

## TODO

按以下顺序推进，每一步先明确接口，再写实现和对应验证。

- [x] 声明逐元素算子、CUDA 启动函数、Tensor 运算符、分配构造函数和只读数据访问接口。
- [ ] 补齐只按 shape、dtype、device 分配存储的 Tensor 接口；明确是否初始化，避免为了分配输出先准备一份 CPU 源数组。
- [x] 实现 `const void* data_ptr() const`，使算子能读取 `const Tensor&` 输入。
- [ ] 完善公共输入检查：拒绝空对象，检查 shape、dtype、device，并拒绝尚未实现的类型或设备。
- [x] 拆分公共入口、CPU 文件与 CUDA 文件，声明可跨文件调用的普通后端入口。
- [x] 将加法循环移入 CPU 文件的 `AddTypedCPU<T>()`，直接遍历连续存储的全部元素。
- [ ] 完成 `add()`：检查输入、一次分配独立输出、按 device/dtype 分发并调用计算函数。
- [ ] 在 `elementwise_cpu.cpp` 的 CPU 入口内接入 `DISPATCH_DTYPE`，选择对应的内部模板实例。
- [x] 将 FP16/BF16 解码函数整理到 `dtype.h`，供算子复用。
- [ ] 实现 CPU 的 FP16/BF16 解码、FP32 计算和编码路径。
- [ ] 明确整数除零和溢出、浮点除零及特殊值处理规则。
- [ ] 声明并实现 `Tensor::operator+`，复用 `add()`。
- [ ] 将 `elementwise.cpp`、`elementwise_cpu.cpp` 和 CPU 测试接入 CMake，覆盖各 dtype、标量、多维输入、索引视图、输入不变、输出独立、特殊值和无效输入。
- [ ] 实现 CUDA 的 `LaunchAddKernel()` 和按 dtype 分发的 kernel，普通数值类型使用模板，FP16/BF16 使用 CUDA 对应的浮点类型；处理线程越界并检查启动错误。
- [ ] 按 `ENABLE_CUDA` 条件接入 CUDA 源码和测试；对照 CPU 结果，覆盖元素数不整除线程块大小的情况，读取结果时检查执行错误。
- [ ] 按相同结构实现 `sub()`、`mul()`、`div()` 及对应运算符，补齐各 dtype 的 CPU/CUDA 测试。
- [ ] 后续独立设计广播、标量混合运算及 `add_()` / `operator+=` 等原地接口。

测试与构建方式见 [tests/README.md](../tests/README.md)。
