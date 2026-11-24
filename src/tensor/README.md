# tensor 设计

**Tensor 管理数据的形状、类型和访问位置。** 当前支持共享存储的 stride 视图，CPU 和 CUDA 的填充、清零、复制、打印及迁移支持非连续视图。

构造、clone、copy_、to、fill_、zero_ 和 print 按 device 分流；未检测到可用 CUDA 编译环境时，CUDA 构造或迁移会抛出异常。

## 文件和字段

| 位置 | 内容 |
| --- | --- |
| `include/mini_infer/core/` | `tensor.h`、`scalar.h`、`dtype.h` |
| `src/tensor/` | 构造、元数据、校验和打印 |
| `src/ops/` | `layout/` 管视图，`copy/` 管复制和迁移，`fill/` 管填充和清零 |
| `src/device/` | CPU/CUDA 内存分配与释放 |

Tensor 拷贝共享同一个 `TensorImpl`；视图创建新 `TensorImpl`，但共享 `Storage` 和写入版本。

| 对象 | 职责 |
| --- | --- |
| `TensorImpl` | 固定容量的 shape、stride，实际维数，dtype，元素偏移，以及按需创建的梯度元数据 |
| `Storage` | 字节容量和 `DataPtr`；最后一个拥有者销毁时释放数据 |
| `DataPtr` | 基地址、设备和对应的释放器 |
| `CUDALayout` | 启动 CUDA kernel 时从宿主布局打包，按值传参，不分配或上传专用显存副本 |

## 内存与索引

布局最多 8 维。创建 Tensor 或视图时校验维数和范围。stride 与偏移的单位都是元素，`data_ptr()` 才换算成字节地址：

```text
视图偏移 = 父 Tensor 相对 Storage 的偏移 + 本次操作的增量
数据地址 = Storage 基地址 + 视图偏移 × element_size()
```

例如 `x` 的 shape 为 `{2, 3}`，`x[1]` 的起点偏移为 3；再做 `.narrow(0, 1, 2)`，结果偏移为 4。只改布局的操作不改变起点。视图持有 Storage，即使原 Tensor 先销毁也可以继续访问数据。

## 关键函数设计

| 函数 | 作用 | 要点 |
| --- | --- | --- |
| `Tensor()` | 创建默认空对象 | 数据指针为空，`numel() == 0` |
| `Tensor(data_ptr, shape, dtype, device)` | 从连续的 CPU 源数据构造 | 自己分配并复制；源数据按 dtype 编码，拒绝空指针、负数维度和元素数为 0 的 shape |
| `Tensor(shape, dtype, device)` | 按元数据分配存储，供算子创建输出 | 只分配不初始化；`shape` 为 `{}` 时按标量分配一个元素；拒绝负数维度和元素数为 0 的 shape |
| 拷贝构造 / 拷贝赋值 | 创建或更新共享数据的 Tensor | 共享同一个 `TensorImpl`，不复制布局或数据 |
| 移动构造 / 移动赋值 | 接管已有资源 | 转移 `TensorImpl`，源对象变为空 Tensor，保留 dtype 和 device |
| 析构 | 释放当前对象持有的资源 | 使用默认析构，由智能指针决定何时释放数据 |
| `operator[](i)` | 沿第一维取子 Tensor | 返回共享视图，每次去掉一维；索引必须非负且不越界 |
| `data_ptr()` | 获取当前数据起点 | 返回 `void*`；读写时由调用方按 dtype 转换类型 |
| `clone()` | 复制出独立数据 | 非连续视图复制为连续结果 |
| `to(device)` | 返回指定设备上的 Tensor | 同设备共享存储；跨设备分配并复制，保持 shape 和 dtype，源 Tensor 不变 |
| `to(dtype)` | 在原设备上转换 dtype | 同 dtype 共享存储；转换时按源 stride 读取，返回连续结果 |
| `copy_(src)` | 把元素写入已有 Tensor | CPU 和 CUDA 支持不同 stride；共享存储的重叠或无法判定安全的非连续视图暂时拒绝 |
| `fill_(value)` | 原地填充数值 | 按 dtype 转换输入；FP16/BF16 先编码一次，再填充 16 位编码 |
| `zero_()` | 原地清零 | CPU 和 CUDA 按 stride 清零逻辑元素 |
| `print()` | 打印内容 | 根据 shape 递归输出括号，FP16/BF16 解码后打印 |
| `narrow(dim, start, length)` | 沿指定维度取范围 | 返回共享存储的视图，保留 stride；当前要求 length 大于 0 |
| `reshape` / `transpose` / `permute` | 形状与布局操作 | reshape 支持一个 -1 推导；连续输入共享存储，非连续输入先复制；transpose、permute 返回共享视图 |
| `contiguous` / `squeeze` / `unsqueeze` | 形状与布局操作 | 已实现；squeeze/unsqueeze 支持非连续视图及反向，目前只接受非负维度 |

`shape()` 和 `stride()` 返回借用的只读布局视图，不能比对应的布局活得更久；要修改或单独保存布局，用 `.to_vector()`。
`size(dim)` 返回指定维度长度，`dim()` 返回维数，`numel()` 返回元素总数，`nbytes()` 返回字节数。`element_size()`、`dtype()`、`device()` 提供对应的元数据。

### 空对象与标量

两者的 shape 都可以是 `{}`，用是否持有数据区分：

| 对象 | 数据指针 | `empty()` | `numel()` | 打印 |
| --- | --- | --- | --- | --- |
| 默认空 Tensor | 空 | `true` | 0 | `tensor([])` |
| 标量 / 标量视图 | 非空 | `false` | 1 | 如 `tensor(6)` |

`a[1][2]` 从二维 Tensor 取到一个零维标量 Tensor，继续对标量使用整数索引会报错。也可以用一个变量的地址和 shape `{}` 构造标量，构造函数会复制该变量的内容。

`Tensor(1.5f)` 创建 CPU Float32 标量，`Tensor(42)` 创建 CPU Int64 标量。超出 Int64 范围的无符号值会抛 `std::out_of_range`。

### 填充

- `fill_(Scalar)` 按目标 dtype 转换数值。浮点转整数向零截断，超出范围或收到 NaN/Inf 时抛 `std::out_of_range`。`Scalar` 分别保存有符号整数、无符号整数和浮点，避免大整数经过浮点舍入。
- `fill_(Tensor)` 接收同 dtype、同设备的零维标量。若两者共享存储，先保存填充值。`full()` 支持用普通数值或零维标量创建并填充 Tensor。[填充实现](../ops/fill/fill.cpp)

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

`a = a.to(...)` 在迁移成功后才替换句柄。原存储由其余 Tensor 或视图继续持有，最后一个引用释放时调用原设备的释放函数。

跨设备迁移先在源设备上整理非连续视图，再复制到目标设备并保留 dtype。空对象迁移只更新 device 元数据。CUDA 迁移要求构建时启用 CUDA。`copy_()` 用于同设备复制，`clone()` 在当前设备上深拷贝。

### dtype 转换

`x.to(Dtype::kFloat32)` 保留设备和 shape。当前支持 FP16/BF16 与 FP32 双向转换，以及 Int8、UInt8、Int16、Int32、Int64 转 FP32。浮点转换的反向把梯度转回输入 dtype；整数转 FP32 不建图。

## 自动微分

Autograd 的数据结构、反向节点和引擎见 [autograd/README.md](../autograd/README.md)。Tensor 提供以下接口：

| 成员 / 方法 | 作用 | 要点 |
| --- | --- | --- |
| `autograd_meta` | 自动微分状态 | `std::shared_ptr<AutogradMetadata>`；拷贝共享，视图新建 |
| `requires_grad()` / `requires_grad_(bool)` | 查询和设置 | 控制是否保存梯度；RunBackward 仅接受 Float32/Float16/BFloat16 |
| `has_grad()` / `grad()` | 查询梯度 | `meta->grad` 是否有值；无梯度返回默认空 Tensor |
| `zero_grad()` | 清零当前 Tensor 的梯度 | 保留梯度存储和 has_grad 状态；无梯度时不操作，不改变计算图或参数数据 |
| `backward()` | 单元素反向入口 | 上游梯度默认为 1，要求 `numel() == 1` |
| `backward(gradient)` | 指定上游梯度 | 支持非标量输出；gradient 须与输出同 shape/dtype/device，输出须有 grad_node |
| `detach()` | 断开图和梯度 | 共享 Storage 和版本，并使用新的 TensorImpl，不继承梯度历史 |
| `grad_node()` / `set_grad_node()` | 图的连边 | 读写 `meta->grad_node`，它指向产生当前 Tensor 的节点 |
| `accumulate_grad()` | 梯度累加 | 首次克隆梯度，之后用 `add()` 累加 |
| `version()` / `update_version()` | 查询版本 / 版本加一 | 共享存储的 Tensor 共用计数；原地写入自动更新，通过 data_ptr 写入须手动通知 |

拷贝共享 `TensorImpl`。视图和 `detach()` 使用新的 `TensorImpl`，只共享 Storage 和写入版本；梯度元数据按需创建。

## dtype

支持 `Float32`、`Float16`、`BFloat16`、`Int8`、`UInt8`、`Int16`、`Int32`、`Int64`。

`Dtype` 标识运行时类型；`dtype_info()` 返回 `DtypeInfo`，提供名称、单元素字节数和是否量化等信息。`dtype_size()` / `dtype_name()` 保留为便捷查询接口，当前各类型均不是量化格式。单元素字节数只适用于当前定长类型，未来分块量化格式需要扩展存储描述。

`Float16` / `BFloat16` 保存 `uint16_t` 原始编码，支持从 `float` 编码、通过 `from_bits()` 保存已有编码，以及转换回 `float`。编码采用最近值、恰好居中时取偶数的舍入规则；打印仍可直接使用公共解码函数。运行时类型由各调用处的 `switch` 分发。

构建与测试命令见 [tests/README.md](../../tests/README.md)。
