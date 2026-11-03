# tensor 设计

**Tensor 管理一块数据的形状、类型和访问位置。** 当前实现连续的 CPU 和 CUDA 存储；CUDA 路径尚待 GPU 实测。

构造、clone、copy_、fill_、zero_ 和 print 按 device 分流；关闭 CUDA 构建选项时，CUDA 构造会抛出异常。

tensor 要解决的问题是：

1. 我多大？
2. 第一个维度的第 i 块数据在哪？
3. 多个 Tensor 共享数据时，谁负责释放？

所以我们可以思考这几个问题：

| 子问题 | 字段 | 当前做法 |
| --- | --- | --- |
| 数据在哪 / 谁释放 | `std::shared_ptr<void> storage_` | 持有原始分配地址，最后一个持有者销毁时调用 CPU 或 CUDA 释放函数 |
| 从哪开始 | `void* data_ptr_` | 指向当前 Tensor 的数据起点；视图可以指向内存中间 |
| 每一维多大 | `std::vector<int> shape` | 每个 Tensor 自己保存形状，元素按连续顺序排列 |
| 每个元素占几字节 | `Dtype dtype_` | 通过 `dtype_size()` 查询 |
| 数据在哪个设备 | `Device device_` | 区分 CPU 与 CUDA；CUDA 路径待 GPU 实测 |

## 内存与索引

`storage_` 管释放，`data_ptr_` 管位置。索引只调整指针和形状，仍共享原来的内存：

```text
a        shape {2, 3}，data_ptr_ 指向开头 ──┐
b = a    shape {2, 3}，data_ptr_ 指向开头 ──┼→ 共享同一块 storage_
a[1]     shape {3}，data_ptr_ 指向第二行 ───┘
```

对连续 Tensor，第一维索引的地址计算是：

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
| 拷贝构造 / 拷贝赋值 | 创建或更新共享数据的 Tensor | 共享 `storage_`，复制形状、dtype、device 和当前数据指针 |
| 移动构造 / 移动赋值 | 接管已有资源 | 转移 `storage_`，源对象的指针置空、形状清空 |
| 析构 | 释放当前对象持有的资源 | 使用默认析构，由智能指针决定何时释放数据 |
| `operator[](i)` | 沿第一维取子 Tensor | 返回共享视图，每次去掉一维；索引必须非负且不越界 |
| `data_ptr()` | 获取当前数据起点 | 返回 `void*`；读写时由调用方按 dtype 转换类型 |
| `clone()` | 复制出独立数据 | 从当前 `data_ptr_` 开始复制，视图也能单独 clone |
| `copy_(src)` | 把元素写入已有 Tensor | shape、dtype、device 和空状态要一致；相同数据地址直接返回 |
| `fill_(value)` | 原地填充数值 | 按 dtype 转换输入；FP16/BF16 先编码一次，再填充 16 位编码 |
| `zero_()` | 原地清零 | 将当前数据区域的所有字节设为 0 |
| `print()` | 打印内容 | 根据 shape 递归输出括号，FP16/BF16 解码后打印 |

`size()` 返回形状，`size(dim)` 返回指定维度长度，`dim()` 返回维数，`numel()` 返回元素总数，`nbytes()` 返回字节数。`element_size()`、`dtype()`、`device()` 提供对应的元数据。

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

## dtype

支持 `Float32`、`Float16`、`BFloat16`、`Int8`、`UInt8`、`Int16`、`Int32`、`Int64`。

`dtype.h` 提供类型枚举、大小与名称查询、FP16/BF16 编码，以及普通 C++ 数值类型的分发宏。编码采用最近值、恰好居中时取偶数的舍入规则；16 位浮点通过 `uint16_t` 保存原始编码。解码位于 `tensor.cpp`，供打印使用。

## todo

### dtype.h

- [x] `Dtype` / `Device` 枚举
- [x] `dtype_size()` / `dtype_name()`
- [x] 普通数值类型分发
- [x] FP16 / BF16 编码与舍入

### tensor.h / tensor.cpp

- [x] CPU 分配、共享所有权与默认析构
- [x] 拷贝与移动语义
- [x] 元数据查询、默认空对象与标量
- [x] 第一维整数索引与多级索引视图
- [x] `clone()` / `copy_()` / `fill_()` / `zero_()`
- [x] 嵌套打印与 FP16 / BF16 解码
- [x] CUDA 内存分配、释放、数据复制和 fill kernel 已写入源码
- [ ] 在 GPU 上编译并运行 CUDA 测试

### 测试

- [x] CPU Tensor 接口、共享内存和视图生命周期
- [x] FP16 / BF16 编码、舍入边界和打印解码
- [x] CMake / CTest 接入，GCC 构建下两组测试通过

构建与测试命令见 [tests/README.md](../tests/README.md)。
