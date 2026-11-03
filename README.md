# mini_infer

从零手写一个大模型推理框架，在实现 Tensor、算子、模型和运行时的过程中，学习和理解现代推理框架如何工作。

## 项目结构

```text
mini_infer/
├── tensor/          # Tensor、dtype、FP16/BF16 编解码
├── utils/           # 内存分配、日志等公共工具
├── tests/           # CPU Tensor 与浮点编码测试
├── CMakeLists.txt   # C++17 构建与 CTest 配置
└── README.md
```

各模块的具体说明见各自目录下的 README.md。

## CUDA 开发基线

- GPU：Ampere 及更新架构（SM80 起）。
- CUDA Toolkit：暂定 12.4；具体 GPU 仍需匹配工具链支持的编译目标。
- Colab L4（CUDA Toolkit 13.0.88）已完成 CUDA 构建，`setup.ipynb` 保存的 4 项 CTest 均通过；Windows RTX 4060 尚待验证。

## Roadmap

- [x] **CPU Tensor**：内存与共享视图、拷贝与移动、填充与复制、打印、FP16/BF16 编解码；两组测试通过。
- [ ] **CUDA Tensor**：补齐 GPU 内存分配与释放、CPU/GPU 数据传输，以及 Tensor 的复制、填充、清零和索引视图支持。
- [ ] **基础算子**：先实现矩阵乘、逐元素运算、归约和 Softmax 等 CPU 参考实现，再补对应的 CUDA 实现并对照验证。
- [ ] **Module / Layer**：组织参数、子模块和 forward，实现 Linear、Embedding、Norm、激活等基础层。
- [ ] **模型前向**：组合 Attention、位置编码和 FFN，接入模型配置与权重加载，跑通一个小型 Decoder-only 模型。
- [ ] **单请求生成**：接入 tokenizer、采样和停止条件，区分 prefill / decode，加入基础 KV Cache，完成文本到文本的生成。
- [ ] **推理运行时**：管理请求、调度 prefill / decode、实现 continuous batching，并完善 KV Cache 的分配与回收。
- [ ] **性能与扩展**：测量延迟、吞吐和显存占用，再选择算子融合、分页 KV Cache、量化或并行等优化方向。

Layer、Block 和 Model 都可以通过 Module 统一组织：算子负责计算，Module 负责参数和组合关系。每个阶段做到时再细化接口和任务，整体保持 WIP。
