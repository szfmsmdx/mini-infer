# mini_infer

从零手写一个训推一体的大模型框架，在实现 Tensor、算子、自动微分、模型和运行时的过程中，学习和理解现代框架如何工作。

## 项目结构

```text
mini_infer/
├── tensor/          # Tensor、dtype、FP16/BF16 编解码
├── ops/             # 逐元素算子的公共入口、CPU/CUDA 后端骨架
├── autograd/        # AutogradMetadata、反向节点与自动微分引擎（WIP）
├── utils/           # 内存分配、日志等公共工具
├── tests/           # CPU/CUDA Tensor、逐元素算子、自动微分、浮点编码与日志测试
├── CMakeLists.txt   # C++17 构建与 CTest 配置
└── README.md
```

各模块的具体说明见各自目录下的 README.md。

## CUDA 开发基线

- GPU：Ampere 及更新架构（SM80 起）。
- CUDA Toolkit：最低 12.4；CMake 自动检测可用 CUDA 编译环境，无需手动切换开关。默认编译 SM80 并保留 PTX，支持更新架构 GPU。
- 当前版本已通过本机 CPU 构建和现有 5 项测试；CUDA 行为需在目标环境用 `test.sh` 重新验证。

## Roadmap

- [x] **CPU Tensor**：内存与共享视图、拷贝与移动、填充与复制、打印、FP16/BF16 编解码；CPU 构建与测试通过。
- [ ] **CUDA Tensor**：支持连续和非连续视图的填充、清零、复制、打印、克隆和迁移；后续完善布局接口和性能。
- [ ] **基础算子**：CPU/CUDA 逐元素算子支持 stride；后续实现矩阵乘、归约和 Softmax。
- [ ] **Autograd / 训练支持**：Tensor 梯度状态、前向建图、反向节点与逆拓扑引擎；先支持逐元素算子，再扩展视图、`clone`/跨设备 `to`、归约和矩阵乘。设计见 [autograd/README.md](autograd/README.md)。
- [ ] **Module / Layer**：组织参数、子模块和 forward，实现 Linear、Embedding、Norm、激活等基础层。
- [ ] **模型前向**：组合 Attention、位置编码和 FFN，接入模型配置与权重加载，跑通一个小型 Decoder-only 模型。
- [ ] **单请求生成**：接入 tokenizer、采样和停止条件，区分 prefill / decode，加入基础 KV Cache，完成文本到文本的生成。
- [ ] **推理运行时**：管理请求、调度 prefill / decode、实现 continuous batching，并完善 KV Cache 的分配与回收。
- [ ] **性能与扩展**：测量延迟、吞吐和显存占用，再选择算子融合、分页 KV Cache、量化或并行等优化方向。

Layer、Block 和 Model 都可以通过 Module 统一组织：算子负责计算，Module 负责参数和组合关系。每个阶段做到时再细化接口和任务，整体保持 WIP。

下一步按 [autograd/README.md](autograd/README.md) 的两日任务清单实现自动微分；视图反向、归约和矩阵乘随算子扩展补齐。
