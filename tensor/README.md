# tensor 设计

**Tensor 管理数据的形状、类型和访问位置。** 当前支持共享存储的 stride 视图，CPU 和 CUDA 的填充、清零、复制、打印及迁移支持非连续视图。

构造、clone、copy_、to、fill_、zero_ 和 print 按 device 分流；未检测到可用 CUDA 编译环境时，CUDA 构造或迁移会抛出异常。

## 文件结构与分发

```text
tensor/
    tensor.h         # Tensor 接口、后端声明、fill_<T> 的值转换
    tensor.cpp       # 元数据、所有权、视图、公共检查和设备分流
    tensor_cpu.cpp   # CPU 复制、清零、填充循环和主机数据打印
    tensor_cuda.cu   # CUDA 复制、清零、fill kernel 及其启动函数
    dtype.h          # dtype/device、类型信息和 FP16/BF16 编解码
    README.md
```

构造、`clone()` 和 `copy_()` 的公共逻辑留在 `tensor.cpp`；实际数据复制交给 CPU/CUDA 后端。源数据构造从 CPU 源数据复制；按 shape 分配存储的构造函数只分配、不初始化。`copy_()` 仍要求两端 device 相同。分配与释放继续使用现有内存工具，`storage_` 和视图的所有权语义保持不变。

`fill_<T>()` 的模板定义留在头文件，检查空状态并将输入值转换为目标 dtype。私有非模板成员 `FillValue()` 在 `tensor.cpp` 中选择设备，再调用 `FillCPU()` 或 `FillCUDA()`。CPU 循环和 CUDA kernel 均在各自的实现文件中。

`print()` 的设备分流留在 `tensor.cpp`。CPU 数据直接交给 `PrintCPU()`；CUDA 数据先复制到主机缓冲区，再交给同一个 `PrintCPU()`。括号、缩进和数值格式化统一放在 `tensor_cpu.cpp`。

tensor 要解决的问题是：

1. 我多大？
2. 第一个维度的第 i 块数据在哪？
3. 多个 Tensor 共享数据时，谁负责释放？

所以我们可以思考这几个问题：

| 子问题 | 字段 | 当前做法 |
| --- | --- | --- |
| 数据在哪 / 谁释放 | `std::shared_ptr<void> storage_` | 持有原始分配地址，最后一个持有者销毁时调用 CPU 或 CUDA 释放函数 |
| 从哪开始 | `void* data_ptr_` | 指向当前 Tensor 的数据起点；视图可以指向内存中间 |
| 每一维多大 | `std::vector<int> shape_` | 每个 Tensor 自己保存形状，通过 `shape()` 查询 |
| 每一维的跨度 | `std::vector<std::size_t> stride_` | 单位是元素个数，通过 `stride()` 查询 |
| 每个元素占几字节 | `Dtype dtype_` | 通过 `dtype_size()` 查询 |
| 数据在哪个设备 | `Device device_` | 区分 CPU 与 CUDA |

## 内存与索引

`storage_` 管释放，`data_ptr_` 管位置。索引只调整指针和形状，仍共享原来的内存：

```text
a        shape {2, 3}，data_ptr_ 指向开头 ──┐
b = a    shape {2, 3}，data_ptr_ 指向开头 ──┼→ 共享同一块 storage_
a[1]     shape {3}，data_ptr_ 指向第二行 ───┘
```

第一维索引的地址计算是：

```text
每块元素数 = numel() / size(0)
字节偏移 = index × 每块元素数 × element_size()
视图地址 = 当前 data_ptr_ 按字节加上偏移
视图形状 = 去掉 shape 的第一维
```

例如 float32、shape `{2, 3}`，第二行的起点在当前地址之后 `1 × 3 × 4 = 12` 字节。原 Tensor 销毁后，视图仍持有 `storage_`，数据继续有效。

## 关键函数设计

public：

| 函数 | 作用 | 要点 |
| --- | --- | --- |
| `Tensor()` | 创建默认空对象 | 数据指针为空，`numel() == 0` |
| `Tensor(data_ptr, shape, dtype, device)` | 从连续的 CPU 源数据构造 | 自己分配并复制；源数据按 dtype 编码，拒绝空指针、负数维度和元素数为 0 的 shape |
| `Tensor(shape, dtype, device)` | 按元数据分配存储，供算子创建输出 | 只分配不初始化；`shape` 为 `{}` 时按标量分配一个元素；拒绝负数维度和元素数为 0 的 shape |
| 拷贝构造 / 拷贝赋值 | 创建或更新共享数据的 Tensor | 共享 `storage_`，复制形状、dtype、device 和当前数据指针 |
| 移动构造 / 移动赋值 | 接管已有资源 | 转移 `storage_`，源对象的指针置空、形状清空 |
| 析构 | 释放当前对象持有的资源 | 使用默认析构，由智能指针决定何时释放数据 |
| `operator[](i)` | 沿第一维取子 Tensor | 返回共享视图，每次去掉一维；索引必须非负且不越界 |
| `data_ptr()` | 获取当前数据起点 | 返回 `void*`；读写时由调用方按 dtype 转换类型 |
| `clone()` | 复制出独立数据 | 非连续视图复制为连续结果 |
| `to(device)` | 返回指定设备上的 Tensor | 同设备共享存储；跨设备分配并复制，保持 shape 和 dtype，源 Tensor 不变 |
| `copy_(src)` | 把元素写入已有 Tensor | CPU 和 CUDA 支持不同 stride；共享存储的重叠或无法判定安全的非连续视图暂时拒绝 |
| `fill_(value)` | 原地填充数值 | 按 dtype 转换输入；FP16/BF16 先编码一次，再填充 16 位编码 |
| `zero_()` | 原地清零 | CPU 和 CUDA 按 stride 清零逻辑元素 |
| `print()` | 打印内容 | 根据 shape 递归输出括号，FP16/BF16 解码后打印 |
| `narrow(dim, start, length)` | 沿指定维度取范围 | 返回共享存储的视图，保留 stride；当前要求 length 大于 0 |
| `reshape` / `transpose` / `permute` / `contiguous` / `squeeze` / `unsqueeze` | 形状与布局操作 | reshape 暂只支持连续输入；其余接口待实现 |

`shape()` 和 `stride()` 返回形状与跨度的只读引用，`size(dim)` 返回指定维度长度，`dim()` 返回维数，`numel()` 返回元素总数，`nbytes()` 返回字节数。`element_size()`、`dtype()`、`device()` 提供对应的元数据。原无参 `size()` 已改为 `shape()`；成员变量使用 `shape_`。

### 空对象与标量

两者的 shape 都可以是 `{}`，用是否持有数据区分：

| 对象 | 数据指针 | `empty()` | `numel()` | 打印 |
| --- | --- | --- | --- | --- |
| 默认空 Tensor | 空 | `true` | 0 | `tensor([])` |
| 标量 / 标量视图 | 非空 | `false` | 1 | 如 `tensor(6)` |

`a[1][2]` 从二维 Tensor 取到一个零维标量 Tensor，继续对标量使用整数索引会报错。也可以用一个变量的地址和 shape `{}` 构造标量，构造函数会复制该变量的内容。

### 共享与复制的区别

```cpp
float data[] = {1, 2, 3, 4, 5, 6};
mtensor::Tensor a(data, {2, 3}, mtensor::Dtype::kFloat32);

mtensor::Tensor b = a;  // 共享数据。
b[0].fill_(9);         // a 的第一行也变成 9。

mtensor::Tensor c = a.clone();
c.zero_();             // c 的数据独立，a 保持原值。

a[1].copy_(a[0]);      // 把第一行的数据写入第二行。
```

赋值运算符更新的是 Tensor 句柄，因此 `a[0] = b` 会给临时视图重新赋值。要修改原数据，使用视图的 `copy_()`、`fill_()` 或 `zero_()`。

### 设备迁移与赋值

```cpp
Tensor gpu = a.to(Device::kCUDA);  // 新分配 GPU 存储并复制，a 仍在 CPU。
Tensor host = gpu.to(Device::kCPU);
Tensor alias = gpu.to(Device::kCUDA);  // 同设备，返回共享存储的句柄。
a = a.to(Device::kCUDA);  // 完成复制后，通过移动赋值让 a 接管结果。
```

`.to()` 本身不改变源对象。赋值时，原 `storage_` 减少一个引用；如果还有其他 Tensor 或视图持有它，原存储继续有效，否则立即调用原设备的释放函数。迁移失败时不会执行赋值，源对象仍持有原数据。

跨设备迁移支持非连续视图，先在源设备上整理成连续副本，再迁移；空对象迁移保持空状态与目标 device，不分配数据。未启用 CUDA 时迁移到 CUDA 会报错。当前 `to()` 仅接收 device，不转换 dtype；`copy_()` 仍只支持同设备复制，`clone()` 在当前设备上深拷贝。

## 自动微分（计划）

Autograd 处于 WIP：已完成 `autograd_metadata_` 的初始化与共享、`requires_grad_()`、`has_grad()`、`grad()`、`grad_node()`、`set_grad_node()`、`accumulate_grad()`、`detach()` 和 `backward()` 入口；`backward()` 已写参数检查，`TopoSort` 和梯度传递待实现。设计与任务拆分见 [autograd/README.md](../autograd/README.md)。Tensor 计划增加：

| 成员 / 方法 | 作用 | 要点 |
| --- | --- | --- |
| `autograd_metadata_` | 自动微分状态 | `std::shared_ptr<AutogradMetadata>`；拷贝共享，视图新建 |
| `requires_grad()` / `requires_grad_(bool)` | 查询和设置 | 控制是否保存该 Tensor 的梯度；dtype 支持范围待反向实现时确定 |
| `has_grad()` / `grad()` | 查询梯度 | `meta->grad` 是否有值；无梯度返回默认空 Tensor |
| `backward()` | 标量反向入口 | 调用 `RunBackward`；第一阶段只支持 `numel() == 1` |
| `detach()` | 断开图和梯度 | 共享 `storage_`，创建新的空 meta |
| `grad_node()` / `set_grad_node()` | 图的连边 | 读写 `meta->grad_node`；Node 只保存输入，不保存输出 |
| `accumulate_grad()` | 梯度累加 | 首次 `grad.emplace(g.clone())`，之后 `add()` |

拷贝共享 `storage_` 和 `autograd_metadata_`；移动后由目标接管这些句柄。视图（`operator[]`、`narrow`）新建 meta，`reshape` 复制 `*this` 后重置 meta。当前在建图开启且 Tensor 参与 autograd（`requires_grad()` 或 `grad_node()` 非空）时，`operator[]`、`narrow`、`reshape`、`clone()` 和跨设备 `to()` 抛 `std::logic_error`；`detach()` 和同设备 `to()` 不受影响。后续实现 `ViewNode`、`CloneNode`、`CopyNode` 后移除保护，使这些操作可微分。

## dtype

支持 `Float32`、`Float16`、`BFloat16`、`Int8`、`UInt8`、`Int16`、`Int32`、`Int64`。

`Dtype` 标识运行时类型；`dtype_info()` 返回 `DtypeInfo`，提供名称、单元素字节数和是否量化等信息。`dtype_size()` / `dtype_name()` 保留为便捷查询接口，当前各类型均不是量化格式。单元素字节数只适用于当前定长类型，未来分块量化格式需要扩展存储描述。

`Float16` / `BFloat16` 保存 `uint16_t` 原始编码，支持从 `float` 编码、通过 `from_bits()` 保存已有编码，以及转换回 `float`。编码采用最近值、恰好居中时取偶数的舍入规则；打印仍可直接使用公共解码函数。运行时类型由各调用处的 `switch` 分发。

## todo

### dtype.h

- [x] `Dtype` / `Device` 枚举
- [x] `DtypeInfo` 与 `dtype_size()` / `dtype_name()`
- [x] 普通数值类型分发
- [x] `Float16` / `BFloat16` 存储类型、编码与舍入
- [x] FP16 / BF16 解码整理为公共 inline 工具

### tensor.h / tensor.cpp

- [x] CPU 分配、共享所有权与默认析构
- [x] 拷贝与移动语义
- [x] 元数据查询、默认空对象与标量
- [x] 第一维整数索引与多级索引视图
- [x] `clone()` / `copy_()` / `fill_()` / `zero_()`
- [x] 按 shape、dtype、device 只分配输出的构造函数
- [x] `narrow` 共享视图与 stride 索引
- [ ] 完善非连续 `reshape`、`transpose` / `permute` / `contiguous` / `squeeze` / `unsqueeze`
- [x] `to(device)`、同设备共享和 CPU/CUDA 双向迁移
- [x] 嵌套打印与 FP16 / BF16 解码
- [x] CUDA 内存分配、释放、数据复制和 fill kernel 已写入源码
- [x] 历史版本在 Colab L4 与 Windows RTX 4060 上编译并运行 CUDA 测试
- [ ] 验证最新 stride 改动的 CUDA 构建和 GPU 行为
- [x] 拆分公共入口、CPU 后端和 CUDA 后端，保持已有接口语义
- [x] `requires_grad_` / `grad()` / `has_grad()` 接口
- [x] `detach()` 接口
- [x] `AutogradMetadata` 与 `autograd_metadata_` 的构造、拷贝/移动和 `reshape` 重置
- [x] `grad_node()` / `set_grad_node()` / `accumulate_grad()`
- [x] `backward()` 标量入口（参数检查已实现，梯度传递待实现）
- [x] 视图、`clone()`、跨设备 `to()` 的临时保护
- [ ] `CloneNode` / `CopyNode` / `ViewNode`，移除保护并支持对应反向
- [ ] 视图反向、归约与矩阵乘的反向支持（后续）

### 测试

- [x] CPU Tensor 接口、共享内存和视图生命周期
- [x] FP16 / BF16 编码、舍入边界和打印解码
- [x] CMake / CTest 接入，GCC 构建下两组测试通过
- [x] `autograd_cpu_test`：建图开关、守卫嵌套与异常恢复、梯度状态共享和移动（助手负责）
- [ ] `autograd_cpu_test`：反向规则、梯度累加、链式/菱形图和有限差分（助手负责）

构建与测试命令见 [tests/README.md](../tests/README.md)。

## stride 工作进度

CPU 和 CUDA 的 `zero_`、`fill_`、`print`、`copy_`、`clone`、跨设备迁移及逐元素算子支持非连续视图。共享存储下无法确定安全的非连续复制仍拒绝。

CUDA Tensor 持有 shape 和 stride 的设备副本；视图创建独立的布局副本，拷贝可共享相同布局。连续快速路径和性能优化可按测量结果推进。
