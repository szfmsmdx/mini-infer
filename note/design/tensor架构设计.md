# Tensor 架构设计

一个 2×3 的 Float32 矩阵，在内存中是六个连续的数。Tensor 需要表达切片视图、跨设备存储、内存复用和自动微分。下面从内存布局出发，说明 Tensor 与运行时、后端、Autograd 之间的边界。

## 一、Tensor 需要表达什么

### 1. 同一排数，怎样读出不同形状？

设矩阵为 `x`，转置为 `t`，第二行为 `r`。三个 Tensor 共用一份底层内存。

```text
                [0, 1, 2, 3, 4, 5]  ← Storage 中的线性元素
                │
┌───────────────┼───────────────┐
▼               ▼               ▼
x               t               r = x[1]
[0, 1, 2]       [0, 3]          [3, 4, 5]
[3, 4, 5]       [1, 4]
                [2, 5]
```

| 对象 | shape | stride（元素） | storage_offset（元素） |
| --- | --- | --- | ---: |
| `x` | `[2, 3]` | `[3, 1]` | 0 |
| `t = x.transpose(0, 1)` | `[3, 2]` | `[1, 3]` | 0 |
| `r = x[1]` | `[3]` | `[1]` | 3 |

stride 表示某维下标加 1 时，在内存中跨过的元素数。访问元素的计算公式：

```text
元素偏移 = storage_offset + Σ(各维下标 × 对应 stride)
字节地址 = Storage 基地址 + 元素偏移 × element_size()
```

创建 Tensor 或视图时，`TensorImpl` 检查排布不越出 `Storage` 容量：

```text
storage_offset + Σ((shape[i] − 1) × stride[i]) < Storage::nbytes() / element_size()
```

### 2. 谁管理底层内存与生命周期？

`Tensor` 只是轻量句柄，共享持有 `TensorImpl`。视图创建新的 `TensorImpl`，但共享同一份 `Storage` 与写入版本号 `version`。

```text
Tensor ─(shared_ptr)─→ TensorImpl
                           ├─ storage_offset / dtype / shape[8] / stride[8]
                           ├─ version_       (shared_ptr<uint64_t>，视图/detach 共享)
                           ├─ autograd_meta_ (shared_ptr<AutogradMetadata>，按需懒创建)
                           └─ storage_       (shared_ptr<Storage>) ──→ DataPtr (基地址 + 设备 + deleter)
```

| 操作 | Storage 去向 | 元数据与版本 |
| --- | --- | --- |
| 拷贝构造 / 赋值 `b = a` | 共享同一个 `TensorImpl` | 共享 `autograd_meta_` 与 `version_` |
| 视图（`[]`/`narrow`/`transpose`/`permute`/`broadcast`/`squeeze`/`unsqueeze`） | 新 `TensorImpl`，共享 `Storage` | 独立 `autograd_meta_`，共享 `version_` |
| `detach()` | 新 `TensorImpl`，共享 `Storage` | 空 `autograd_meta_`，共享 `version_` |
| `clone()` / 跨设备 `to(device)` / `to(dtype)` / 非连续 `reshape()` | 分配独立 `Storage` | 独立 `autograd_meta_` 与新 `version_` |

### 3. 显存管理与分配策略：`Allocator` 与显存池

Tensor 通过 `runtime::GetAllocator(device)` 申请 `Storage`。在 CUDA 推理和训练中，每次构造临时 Tensor 都直接调用 `cudaMalloc` / `cudaFree` 会触发设备同步并降低吞吐。

因此将分配策略收口到 `src/runtime/`：

- **统一接口**：`Allocator::allocate(std::size_t nbytes) -> DataPtr` 位于 `include/mini_infer/tensor/allocator.h`。
- **CPU 分配器**：包装 `std::malloc` / `std::free`。
- **CUDA 分配器与显存池（`CUDACachingAllocator`）**：默认启用按大小分桶的显存块缓存池（256 字节对齐、向上取整到分桶粒度），`DataPtr` 的 deleter 将释放的内存归还给缓存池而非立即 `cudaFree`；显存不足或程序退出时支持 `empty_cache()` 统一回收。
- **`Tensor` 和 `Storage` 不持有 `cudaStream_t`**：流与临时 workspace 统一归 `runtime::DeviceContext` 管理；`Tensor` 只描述数据位置和设备类型。

### 4. 形状怎样交给 GPU？

kernel 运行在设备上，无法读取主机上的 `TensorImpl`。当前沿用按值传参方案：

- 维数上限 `kMaxDim = 8`，由 `ShapeView` / `StrideView` 在 CPU 侧借用只读视图。
- 启动 CUDA kernel 时，由 `src/backend/common/cuda_layout.h` 将 `dim`、`contiguous`、`shape[8]`、`stride[8]` 打包进 POD 结构 `CUDALayout` 按值传入 kernel，不在显存中维护额外的布局副本。

### 5. 原地修改与 `SavedTensor` 版本检查

如果前向节点保存了输入（或复用输出）的引用，而用户在反向前通过任意共享别名执行了 `fill_()`、`zero_()`、`copy_()`，反向读取到的数据就会失真。

- **写前递增**：所有原地写入操作在校验完成后递增 `version_`。
- **统一封装 `SavedTensor`**：替代旧版在各个 `AutogradNode` 里手写 `uint64_t versions[]` 和 `CheckVersion` 的脆弱做法。

```cpp
// include/mini_infer/autograd.h
class SavedTensor {
public:
    SavedTensor() = default;
    explicit SavedTensor(const Tensor& tensor, bool is_output = false)
        : tensor_(is_output ? tensor.detach() : tensor),
          saved_version_(tensor.version()) {}

    Tensor unpack(const char* debug_name = "SavedTensor") const;
};
```

- 保存输出（如 `ExpNode`、`SigmoidNode`、`SoftmaxNode`、`ExtremaNode`）时传入 `is_output = true`，自动执行 `detach()` 切断循环引用并保留版本追踪。
- 反向 `unpack()` 自动比对 `tensor_.version() != saved_version_`，不一致时抛出 `std::logic_error`（包含 `"modified after forward"` 提示）。

## 二、Tensor 与算子、后端、Autograd 的边界

在旧实现中，算子入口、`AutogradNode` 与 CPU/CUDA 后端混在 `src/ops/` 的各个子目录内。重构后职责边界彻底理顺：

```text
用户调用 a + b / a.matmul(b) / a.copy_(b)
  ↓
L3 算子门面 (src/ops/)
  ├─ 1. 校验输入、推导输出元信息 (shape / dtype / device)
  ├─ 2. 分配输出 Tensor（或复用 op_out 传入的目标 Tensor）
  ├─ 3. 通过 L1 分发表 (KernelTable) 按 device 派发到后端纯计算内核
  └─ 4. 若 ShouldRecordGrad(inputs, out.dtype()) 为真，挂载设备无关的 AutogradNode
```

关键约定：

| 层次 | 职责 | 禁止跨越的边界 |
| --- | --- | --- |
| `src/tensor/` | `Tensor` 构造、布局查询、元数据与 `autograd_meta_` 槽位 | 不含 CUDA 语法，不直接写算子计算逻辑 |
| `src/runtime/` | `Allocator`、CUDA 显存池、`DeviceContext`、`CopyBytes`、`CUDA_CHECK` | 不认识算子与 `AutogradNode` |
| `src/ops/` | 对外/对内算子入口、`*_out` 与 `AutogradNode` 定义 | **严禁出现 `#ifdef ENABLE_CUDA` 或直接写 CPU/CUDA 计算细节** |
| `src/backend/` | `KernelTable` 分发及 `cpu/`、`cuda/` 纯计算内核（含前向与反向） | **严禁引用 `autograd.h` 或创建 `AutogradNode`** |
| `src/autograd/` | `GradMode` / `NoGradGuard`、`SavedTensor`、DFS 拓扑反向引擎 | 不感知硬件设备类型 |

## 三、头文件与内部模块结构

```text
include/mini_infer/
├── tensor/
│   ├── device.h          # Device::kCPU / Device::kCUDA
│   ├── dtype.h           # Dtype 枚举、DtypeInfo、FP16/BF16 编解码
│   ├── scalar.h          # 标量参数封装与无损范围转换
│   ├── allocator.h       # DataPtr 与 Allocator 接口
│   ├── storage.h         # Storage 容量与 DataPtr 持有
│   ├── layout_view.h     # kMaxDim、ShapeView、StrideView
│   └── tensor.h          # Tensor 公开句柄
├── runtime/
│   └── memory_pool.h     # 显存池统计与缓存清理接口
└── autograd.h            # AutogradMetadata、SavedTensor、AutogradNode、NoGradGuard、RunBackward

src/
├── tensor/               # tensor_impl.h, tensor.cpp, check.h/.cpp, print.h/.cpp
├── runtime/              # memory.h/.cpp, device_context.h/.cpp, cuda_check.h
├── autograd/             # autograd.cpp
├── ops/                  # layout.cpp, tensor_ops.cpp, elementwise.cpp ... (纯 C++)
└── backend/              # dispatch.h/.cpp + common/ + cpu/ + cuda/
```

类图与依赖关系见 [tensor理想类图.drawio](tensor理想类图.drawio)，全框架五层设计与训推数据流见 [框架架构设计.md](框架架构设计.md)。
