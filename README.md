# mini_infer

从零手写一个训推一体的大模型框架，在实现 Tensor、算子、自动微分、模型和运行时的过程中，学习和理解现代框架如何工作。

## 项目结构

```text
mini_infer/
├── tensor/          # Tensor 状态、存储所有权、dtype 与 Scalar
├── ops/             # Tensor 操作、数值算子和 CPU/CUDA kernel
├── autograd/        # 梯度状态、反向节点与自动微分引擎（WIP）
├── utils/           # 内存分配、日志等公共工具
├── tests/           # CPU/CUDA Tensor、逐元素算子、自动微分、浮点编码与日志测试
├── CMakeLists.txt   # C++17 构建与 CTest 配置
└── README.md
```

各模块的具体说明见各自目录下的 README.md。

## CUDA 开发基线

- GPU：Ampere 及更新架构（SM80 起）。
- CUDA Toolkit：最低 12.4；CMake 自动检测可用 CUDA 编译环境，无需手动切换开关。默认编译 SM80 并保留 PTX，支持更新架构 GPU。
- 构建与测试命令见 [tests/README.md](tests/README.md)。

## Roadmap

- [x] **Tensor 基础**：存储与共享视图、标量、dtype、填充、复制和设备迁移。
- [x] **CPU 计算与反向**：逐元素算子、全元素 `sum`、反向引擎及对应 CPU 测试。
- [x] **布局操作**：`transpose`、`permute` 的前向和反向。
- [ ] **CUDA 计算与反向**：设备迁移、逐元素算子和全元素 `sum` 的最新实现及测试需要 GPU 验证。
- [ ] **视图反向**：`operator[]`、`narrow`、`reshape`。
- [ ] **基础算子**：广播、指定维度归约、`mean`、矩阵乘和 Softmax。
- [ ] **Module / Layer**：组织参数、子模块和 forward，实现 Linear、Embedding、Norm、激活等基础层。
- [ ] **模型前向**：组合 Attention、位置编码和 FFN，接入模型配置与权重加载，跑通一个小型 Decoder-only 模型。
- [ ] **单请求生成**：接入 tokenizer、采样和停止条件，区分 prefill / decode，加入基础 KV Cache，完成文本到文本的生成。
- [ ] **推理运行时**：管理请求、调度 prefill / decode、实现 continuous batching，并完善 KV Cache 的分配与回收。
- [ ] **性能与扩展**：测量延迟、吞吐和显存占用，再选择算子融合、分页 KV Cache、量化或并行等优化方向。

Layer、Block 和 Model 可以通过 Module 统一组织：算子负责计算，Module 负责参数和组合关系。
