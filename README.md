# mini_infer

从零手写一个训推一体的大模型框架，在实现 Tensor、算子、自动微分、模型和运行时的过程中，学习和理解现代框架如何工作。

## 项目结构

```text
mini_infer/
├── include/mini_infer/  # Tensor、dtype、Scalar、Autograd、算子和网络模块的公开头文件
├── src/
│   ├── tensor/          # Tensor 存储所有权、元数据与内部校验
│   ├── ops/             # 按算子类别组织入口、反向节点与 CPU/CUDA 实现
│   ├── modules/         # Module、Linear 和 RMSNorm
│   ├── layers/          # Embedding 和 SimpleDecoderLayer
│   ├── init/            # 参数初始化所用的随机采样
│   ├── models/          # GPT 和 checkpoint
│   ├── tokenizer/       # 字节 tokenizer 和 BPE
│   ├── optim/           # SGD、Adam 的参数更新与优化器状态
│   ├── autograd/        # 建图开关与反向引擎
│   ├── device/          # 内存分配与释放
│   └── utils/           # 日志
├── tests/               # CPU/CUDA 正确性测试与独立 benchmark
├── examples/nanogpt/    # 字符级 GPT 的训练、续写和语料准备
├── 3rdparty/            # benchmark 使用的 ggml
├── CMakeLists.txt       # C++17 库目标与构建选项
├── setup.ipynb          # Colab 构建、测试和 benchmark
└── README.md
```

链接 CMake 目标 `mini_infer` 后，可通过 `#include "mini_infer/core/tensor.h"` 使用 Tensor。算子后端与具体反向节点的头文件位于 `src/`，供库内部使用。

模块说明见 [Tensor](src/tensor/README.md)、[算子](src/ops/README.md)、[网络模块](src/modules/README.md)、[Autograd](src/autograd/README.md) 和 [测试](tests/README.md)。

正确性测试覆盖 CPU 和 CUDA。逐元素和 RMSNorm 精度测试对比 FP64 数学参考及 PyTorch。[测试说明](tests/README.md#数值精度检查) · [精度排查记录](note/code/算子精度问题.md)

SGD 和 Adam 支持 CPU/CUDA 的 FP32 参数。模型和优化器共享权重 Tensor，`step()` 原地更新权重。训练循环依次调用 `zero_grad()`、前向与 loss、`backward()`、`step()`。[优化器用法](tests/README.md#优化器测试)

## CUDA 开发基线

- GPU：Ampere 及更新架构（SM80 起）。
- CUDA Toolkit 最低版本为 12.4。CMake 自动检测 CUDA 编译环境，默认编译 SM80 并保留 PTX，支持更新架构的 GPU。
- 构建与测试命令见 [tests/README.md](tests/README.md)。

## 字符级 GPT 示例

字符级 GPT 的语料准备、CUDA 训练和 checkpoint 加载命令见 [训练和加载字符级 GPT](examples/nanogpt/README.md)。

## 项目进度

| 已实现 | 内容 |
| --- | --- |
| Tensor | 共享存储、stride 视图、复制、设备迁移和常用视图的反向 |
| 算子 | 逐元素、广播、`sum`、`mean`、矩阵乘、RMSNorm、词嵌入、RoPE、因果注意力和交叉熵 |
| 训练 | Autograd、SGD、Adam、Tiny Shakespeare 上的小型 GPT 训练 |
| 生成 | 最终 checkpoint 保存和加载，字节级 GPT 续写 |
| Tokenizer | 字节 tokenizer 用于 GPT，BPE 已独立实现 |

### 待办

- [ ] 补独立的 Softmax 等基础算子，将 BPE 接入 GPT。
- [ ] 推理区分 prefill 和 decode，加入 KV Cache，再实现请求调度和 continuous batching。
- [ ] 测量延迟、吞吐和显存占用，再按结果做算子融合、量化和并行。
