# Tensor 架构设计

一个 2×3 的 Float32 矩阵，在内存里是六个连续的数。我们要用 C++ 表达它，还要支持切片、GPU 计算和反向传播。从读出第一个元素开始，逐步看 Tensor 需要哪些部件。

文中用 PyTorch 风格的写法表示操作。`x[i, j]` 指第 `i` 行、第 `j` 列的元素。

## 一、Tensor 需要表达什么

### 同一排数，怎样读出不同形状？

设矩阵为 `x`，它的转置为 `t`，第二行为 `r`。三个 Tensor 共用一排数据。

```text
                [0, 1, 2, 3, 4, 5]  ← 内存中的顺序
                │
┌───────────────┼───────────────┐
▼               ▼               ▼
x               t               r = x[1]
[0, 1, 2]       [0, 3]          [3, 4, 5]
[3, 4, 5]       [1, 4]
                [2, 5]
```

| 对象 | shape | stride（元素） | 距底层内存起点的偏移（元素） |
| --- | --- | --- | ---: |
| `x` | `[2, 3]` | `[3, 1]` | 0 |
| `t = x.transpose(0, 1)` | `[3, 2]` | `[1, 3]` | 0 |
| `r = x[1]` | `[3]` | `[1]` | 3 |

stride 是某一维的下标增加 1 时，要在内存中跨过多少个元素。对 `x`，换行跨 3 个元素，换列跨 1 个。转置交换这两个步长。索引从 0 开始，`r` 的偏移为 3，所以它从底层内存的第四个元素开始读。

```text
元素位置 = 起始偏移 + Σ(各维下标 × 对应 stride)

t[2, 1]  →  元素位置 = 0 + 2 × 1 + 1 × 3 = 5
         →  字节偏移 = 5 × 4 = 20，读底层内存起始地址 + 20 字节
         →  从该地址读出 Float32 数值 5
```

像 `t` 和 `r` 这样通过自己的布局访问同一块数据的 Tensor，叫作 **视图** （view）。 [PyTorch Tensor Views](https://docs.pytorch.org/docs/2.15/tensor_view.html)

### 谁保证指向的内存还有效？

销毁 `x` 只会释放 `x` 持有的引用。只要 `t` 或 `r` 仍引用同一份 **Storage** 对象，这块内存就继续有效。

```text
x ─┐
t ─┼── 共享 ──→ Storage（字节容量）── 持有 ──→ DataPtr（基地址、设备、释放方式）
r ─┘
```

| 操作 | 结果使用哪份 Storage？ | 数据怎么处理？ |
| --- | --- | --- |
| 从 `x` 转置得到 `t`，或切片得到 `r` | 和 `x` 共用原 Storage | 结果按自己的布局访问原数据。 |
| `t.clone()` | 为结果分配新 Storage | 按 `t` 的布局从原 Storage 读取元素，复制到结果的 Storage。 |
| `c = t.contiguous()` | 为 `c` 分配新 Storage | 按 `t` 的行列顺序读原 Storage，把 `[0, 3, 1, 4, 2, 5]` 连续写到 `c` 的 Storage。 |
| `x.contiguous()` | 复用 `x` 的原 Storage | `x` 已经连续，直接返回 `x`。 |

`t` 和 `c` 的 shape 都是 `[3, 2]`。它们的 stride 分别是 `[1, 3]` 和 `[2, 1]`。 [PyTorch Tensor Views](https://docs.pytorch.org/docs/2.15/tensor_view.html)

### 搬到 GPU 之后，计算怎样发生？

假设 `x` 在 CPU 上，六个 Float32 元素共占 24 字节。

| 概念 | 它回答的问题 | 用 `x` 做什么 |
| --- | --- | --- |
| Device | 数据在 CPU 上，还是在哪张 GPU 上？ | `g = x.to(cuda)` 从 `x` 的 CPU Storage 读取 24 字节的 Float32 数据，复制到 `g` 的新 GPU Storage。 |
| Dtype | 元素怎么编码？ | `h = x.to(Float16)` 读出 `x` 的 Float32 元素，转成 Float16 后写入 `h` 的新 CPU Storage，数据占 12 字节。 |
| Scalar | 传给算子的是哪个数？ | 在 `g.fill_(2)` 中，Scalar 保存传进来的数值 `2`。 |
| **Allocator（分配器）** | 怎么申请内存？ | 为 `g` 在 GPU 上申请 24 字节。分配器返回带释放方式的 DataPtr，交给 Storage 持有。 |

如果把一个数当成 Tensor 使用，它的 shape 就是 `[]`。这种零维 Tensor 也有 Storage、dtype 和 device，浮点零维 Tensor 可以参与反向传播。

```text
g.fill_(2) → Scalar 提供数值 2 → 按 g.dtype() 转成 Float32
           → 按 g.device() 选择 CUDA 实现 → 提交任务，把值写进 g 的 Storage
```

CUDA 的 **Stream** 是 GPU 的任务队列，同一队列里的任务按顺序执行。CPU 提交异步任务后可以先做别的事。

| 时间 → | ① 提交 | ② CPU 先走一步 | ③ GPU 完成 |
| --- | --- | --- | --- |
| CPU | 调用 `g.fill_(2)` | 销毁 `g`，释放这份 Storage 的最后一个 C++ 引用 | 继续执行 |
| GPU stream | 写入任务入队 | kernel 仍通过设备地址访问这块显存 | 写入结束 |

如果显存池在 ② 就把同一个地址交给另一条 stream 写，两项任务可能同时改写它。复用时，运行时要保证新任务在旧任务用完这块显存之后才访问它。同一条 stream 可以依靠队列顺序，跨 stream 需要跟踪使用并安排等待。 [PyTorch CUDA 流与内存管理](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-streams)

### 反向传播为什么要记住前向？

#### 先手算一个位置的梯度

取一个和 `t` 同形状的 `w`，只对 `w` 求梯度。计算 `loss = (t * w).sum()` 时，先把两个矩阵的对应元素相乘，再把乘积加起来。

```text
p = t * w
p[0, 1] = t[0, 1] × w[0, 1] = 3 × w[0, 1]
loss = 其他项之和 + 3 × w[0, 1]
```

固定其他项，`w[0, 1]` 每增加 1，`loss` 就增加 3。所以 `loss` 对这个位置的梯度是 3。

#### 反向节点需要保存什么？

Autograd 在前向时为乘法和求和各记录一个节点。反向时，每个节点根据收到的输出梯度计算输入的梯度。

| 反向顺序 | 这个位置的梯度怎么算？ |
| --- | --- |
| 从 `loss` 回到 `p[0, 1]` | 这一项在求和里加了一次，梯度是 `1`。 |
| 从 `p[0, 1]` 回到 `w[0, 1]` | 把收到的 `1` 乘以前向时的 `t[0, 1]`，得到 `1 × 3 = 3`。 |

乘法节点需要的是前向时 `t` 的值。它可以保存一个 Tensor 引用，沿用 `t` 的布局并持有同一份 Storage，省下整块数据的复制。

#### 保存引用后，谁来检查数据有没有变？

`x[1, 0]` 和 `t[0, 1]` 指向同一个元素。前向结束后，如果通过 `x` 把那个元素写成 30，乘法节点沿着保存的引用读 `t`，就会看到 30。

为识别这种修改，`x` 和 `t` 共用一个版本计数。原地写入会推进计数，乘法节点手里还留着前向时记下的版本。

| 时刻 | `t[0, 1]` | `x` 和 `t` 的当前版本 | 乘法节点记下的版本 |
| --- | ---: | --- | --- |
| 前向 | 3 | `v` | `v` |
| 写入 `x[1, 0] = 30` 后 | 30 | `v + 1` | `v` |

反向时先比较版本。当前是 `v + 1`，前向记下的是 `v`。节点发现版本不符就报错，避免算出错误的梯度 `1 × 30`。 [SavedVariable 版本检查](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/saved_variable.cpp)

这个例子只对 `w` 求梯度，但 `t` 的数据参与了梯度计算。这就是为什么供反向读取的 Tensor 即使自身无需梯度，也要能追踪版本。 [PyTorch 版本共享说明](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h#:~:text=Version%20Counter%20Sharing)

## 二、这些能力由哪些模块实现？

`core/` 放全库共用的基础类型。以 `Scalar` 为例，它自身只负责保存和转换数值，Tensor 和算子都能用它接收参数。Tensor 专用的构造、索引和视图实现放在 `src/tensor/`。

用 `t * w` 串起各模块。下图的箭头表示一方要读取什么，或把哪项工作交给谁。

```text
算子入口 ──读取布局、准备输出──→ Tensor → TensorImpl ──共享──→ Storage
算子入口 ──按设备选择实现──────→ CPU/CUDA 后端
算子入口 ──按需挂反向节点──────→ Autograd ──保存输入引用，记录版本──→ Tensor

设备运行时 ──选择分配器──→ Allocator ──返回──→ DataPtr ──交给──→ Storage
CUDA 后端 ──通过设备运行时提交任务──→ 当前 Stream
```

- Tensor 公开调用接口。TensorImpl 保存 shape、stride、偏移、dtype 和版本状态。同一份 Storage 可以被多个布局不同的 TensorImpl 引用。
- 设备运行时选设备和分配器，再按字节数申请内存。这 24 字节按 `[2, 3]` 还是 `[3, 2]` 访问，由 Tensor 的布局决定。换成显存池时，视图规则仍能沿用。
- `t * w` 的入口为结果分配 Storage。后端拿到输入和输出各自的首元素地址、shape、stride 和 dtype，从输入读取并写到结果。首元素地址已包含视图偏移。
- 反向节点的公式随算子放。Autograd 引擎负责遍历计算图和累加梯度。

### PyTorch 把这些边界放在哪里？

| 从哪个问题追进去 | PyTorch 的分工 | 源码或文档 |
| --- | --- | --- |
| 调用者拿的是什么？ | ATen 的 `TensorBase` 持有 `c10/core` 里的 `TensorImpl`。后者记录 shape、stride 和偏移，并引用存储。 | [TensorBase](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/TensorBase.h) · [TensorImpl](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h) |
| 谁持有数据？ | `StorageImpl` 记录容量并持有 `DataPtr`，后者关联地址、设备和释放方式。 | [StorageImpl](https://github.com/pytorch/pytorch/blob/main/c10/core/StorageImpl.h) · [DataPtr](https://github.com/pytorch/pytorch/blob/main/c10/core/Allocator.h) |
| 谁检查前向值有没有被改？ | 版本计数位于 `TensorImpl`，梯度元数据通过 `AutogradMetaInterface` 接入。保存的 Tensor 在反向取出时核对版本。 | [TensorImpl](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h) · [SavedVariable](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/saved_variable.cpp) |
| 谁决定运行哪段代码？ | Dispatcher 根据 Tensor 的 dispatch keys 和线程局部状态选择处理路径。CPU、CUDA 是后端，Autograd、Autocast 也可以参与调用链。 | [PyTorch Dispatcher](https://docs.pytorch.org/devlogs/dispatcher/2026-04-16-how-does-the-dispatcher-work/) |

对于只有 CPU 和 CUDA 的小框架，可以先由算子入口选择后端。等多个入口要共享同一套选择规则，再考虑提取分发层。

## 三、依赖关系怎样变成目录？

`include/mini_infer/core/` 放共用类型和接口。`src/` 放 Tensor 的内部实现，以及 CPU 和 CUDA 各自的分配与执行代码。下面的目标结构只列 Tensor 相关的部分。

```text
include/mini_infer/
├── core/
│   ├── device.h
│   ├── dtype.h
│   ├── scalar.h            # 算子的数值参数
│   ├── allocator.h         # Allocator 和 DataPtr
│   ├── storage.h           # 共享内存的所有权
│   ├── stream.h            # GPU 执行队列
│   └── tensor.h            # 用户调用的 Tensor 接口
├── autograd.h
└── ops/
    └── elementwise.h

src/
├── tensor/
│   ├── tensor_impl.h        # Tensor 的布局、类型和版本状态
│   ├── tensor.cpp           # 构造、布局查询、存储关联
│   └── layout.cpp           # 索引、切片、转置等视图语义
├── device/
│   ├── cpu_allocator.cpp
│   ├── cuda_allocator.cpp
│   └── cuda_stream.cpp
├── ops/
│   └── elementwise/
│       ├── elementwise.cpp  # 校验、分派、对应的反向规则
│       ├── kernels.h        # 入口与后端之间的私有声明
│       ├── cpu.cpp
│       └── cuda.cu
└── autograd/
    └── autograd.cpp         # 图遍历、梯度累加
```

顺着调用看文件。`x.transpose(...)` 的布局规则在 `src/tensor/`，内存分配和 CUDA stream 管理在 `src/device/`，计算与对应的求导规则在 `src/ops/`，反向图的执行在 `src/autograd/`。
