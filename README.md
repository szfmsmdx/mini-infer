# mini_infer

从零手写一个训推一体的大模型框架，在实现 Tensor、算子、自动微分、模型和运行时的过程中，学习和理解现代框架如何工作。

## 项目结构

```text
mini_infer/
├── include/mini_infer/  # Tensor、dtype、Scalar、Autograd、算子和网络模块的公开头文件
├── src/
│   ├── tensor/          # Tensor 存储所有权、元数据与内部校验
│   ├── ops/             # 布局、填充、复制、数值算子与 CPU/CUDA kernel
│   ├── modules/         # 使用现有算子组合网络模块，例如 Linear
│   ├── autograd/        # 反向引擎与反向节点
│   └── utils/           # 内存分配与日志
├── tests/               # CPU/CUDA 正确性测试与独立 benchmark
├── 3rdparty/            # 按需构建 benchmark 使用的 ggml
├── CMakeLists.txt       # C++17 库目标与构建选项
├── setup.ipynb          # Colab 构建、测试和 benchmark
└── README.md
```

链接 CMake 目标 `mini_infer` 后，可通过 `#include "mini_infer/tensor.h"` 使用 Tensor。算子后端与具体反向节点的头文件位于 `src/`，供库内部使用。

模块说明见 [Tensor](src/tensor/README.md)、[算子](src/ops/README.md)、[网络模块](src/modules/README.md)、[Autograd](src/autograd/README.md) 和 [测试](tests/README.md)。

逐元素精度测试采用双参考：逐元素与数学参考误差达标，或者与 PyTorch 对齐，满足其一即通过。基础、边界和特殊值样本必须全部通过；随机样本按算子、dtype、前向及各输入梯度分别要求至少 99.5% 通过。测试直接使用环境中已有的 PyTorch，不限制版本，并记录实际版本。本地 CPU 当前覆盖的 10 个逐元素算子、三种浮点 dtype 的前向与反向，884,818 次双参考比较全部通过。CUDA 尚未验证。安装、复跑和结果见[数值精度检查](tests/README.md#数值精度检查)。

RMSNorm 已实现组合前向和自动反传，但新增 CPU 精度测试发现 FP16/BF16 共 84 个失败，结构测试通过，低精度问题待修复。结果见[测试说明](tests/README.md#rmsnorm-组合算子测试)；本轮排查过程记录在[算子精度问题](note/code/算子精度问题.md)。

## CUDA 开发基线

- GPU：Ampere 及更新架构（SM80 起）。
- CUDA Toolkit：最低 12.4；CMake 自动检测可用 CUDA 编译环境，无需手动切换开关。默认编译 SM80 并保留 PTX，支持更新架构 GPU。
- 构建与测试命令见 [tests/README.md](tests/README.md)。

## Roadmap

- [x] **Tensor 基础**：存储与共享视图、标量、dtype、填充、复制和设备迁移。
- [x] **CPU 计算与反向**：逐元素算子、全元素 `sum`、反向引擎及对应 CPU 测试。
- [x] **布局操作**：`transpose`、`permute` 的前向和反向。
- [x] **CUDA 计算与反向**：设备迁移、逐元素算子和全元素 `sum` 的测试在 Colab A100 上通过。
- [x] **视图反向**：`operator[]`、`reshape`、`narrow` 已实现，本地 CPU 及 Colab CPU/CUDA 回归已通过。
- [ ] **基础算子**：广播、指定维度归约、`mean`、矩阵乘和 Softmax。
- [ ] **Module / Layer**：组织参数、子模块和 forward，实现 Linear、Embedding、Norm、激活等基础层。
- [ ] **模型前向**：组合 Attention、位置编码和 FFN，接入模型配置与权重加载，跑通一个小型 Decoder-only 模型。
- [ ] **单请求生成**：接入 tokenizer、采样和停止条件，区分 prefill / decode，加入基础 KV Cache，完成文本到文本的生成。
- [ ] **推理运行时**：管理请求、调度 prefill / decode、实现 continuous batching，并完善 KV Cache 的分配与回收。
- [ ] **性能与扩展**：测量延迟、吞吐和显存占用，再选择算子融合、分页 KV Cache、量化或并行等优化方向。

Layer、Block 和 Model 可以通过 Module 统一组织：算子负责计算，Module 负责参数和组合关系。
