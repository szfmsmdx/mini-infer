# mini_infer 项目说明（供后续 AI 会话读取）

## 目标与协作方式

- 这是为了学习现代大模型推理框架而从零手写的 C++ 项目；按实际需要实现 Tensor、CUDA、算子、模型和运行时，不追求完整对齐 PyTorch。
- 用户自己参与接口设计和实现，仍在学习 C++。解释新语法时用当前代码举小例子，说明数据所有权、调用方式和为什么这样写；用户要自己写的部分以引导和审查为主。
- 优先选容易读、容易验证的实现。不要为了抽象而新增包装层、宏、模板或文件；不要随意改已有命名、注释和接口语义。用户明确让助手实现时再直接修改。
- **单测和回归由用户发起。** 每次按建议改完代码，先汇报改动。等用户明确要求测试，再把这批改动一起验证。
- 中文技术说明用短句，按代码、数据流和可核对的结果来写。使用 `concise-tech-docs-zh` 和 `shuorenhua` 时，先读取对应的 `SKILL.md`。
- 新会话开始先看 `git status --short` 和相关源码。区分“本地工作区已修改”和“已推送到 GitHub”，不要覆盖用户的未提交改动；提交或推送时遵守用户当次要求。

## Git 提交规则（必须遵守）

- 提交标题统一使用 `[类型] 概述` 格式，重构使用 `[refact] 概述`。标题下空一行，正文用 `1.`、`2.`、`3.` 有序列表逐条说明本次新增的功能、完成的调整、验证结果和仍未完成的部分；不要只写一个类型名或一段流水账。示例：

  ```text
  [feat] 实现 CUDA 逐元素算子

  1. 实现 add/sub/mul/div/neg 的 CUDA kernel，覆盖 8 种 dtype。
  2. 新增 elementwise_cuda_test，与 CPU 结果逐字节对照。
  3. 在 CMake 注册测试并更新文档。
  ```

- 当前任务或独立功能完成，并通过适用的验证后，可以自主提交和推送；小接口、小功能和文档修改也适用。完成与否按本次任务范围判断，不要求整个项目或其他独立功能全部完成。
- 当前任务尚未完成、必要验证未通过或未能执行时，不提交或推送该任务的改动，除非用户明确要求提交或推送当前快照。按用户要求提交未完成的功能时，标题在类型前缀后标注 `WIP:`，正文说明未完成项和验证情况。
- 一个功能的实现、修复、测试和文档尽量放在同一条提交。已有 WIP 提交且 `HEAD` 属于同一功能时，可在核对提交范围后 amend；功能完成时更新标题和正文，准确反映最终内容。独立的已完成任务可新建提交，不必修改其他功能的 WIP 提交。
- 提交前检查暂存内容，不混入其他功能或用户未授权的本地改动。改写已推送的提交时先核对远端状态，并使用 `--force-with-lease`；不要使用无保护的 `--force`。
- 用户当次明确指定的提交范围、格式和方式优先；任务是否完成或待提交范围无法判断时，先向用户确认。

## C/C++ include 顺序

- 适用于项目中的 C/C++ 头文件和源文件，包括后续的 `.cu` / `.cuh`。
- 先集中放系统、标准库和第三方头文件，使用 `#include <...>`；然后空一行，集中放项目自己的头文件，使用 `#include "..."`。两个组内不额外空行。
- 项目头文件按实际需要和依赖顺序排列；`.cpp` 等实现文件对应的头文件放在项目头文件组的最前面。不要把它单独放到系统头文件前面。
- `#pragma once` 保持在头文件开头，与 include 区域空一行。整理顺序时不要改动接口或实现逻辑。

## 当前设计

| 目录 | 职责 |
| --- | --- |
| `include/mini_infer/` | 公开头文件 |
| `src/tensor/`、`src/device/` | Tensor 元数据、存储所有权、内存分配和释放 |
| `src/ops/` | 按类别组织算子入口、反向节点和 CPU/CUDA 实现。因果注意力、交叉熵、RoPE 分别在 `attention/`、`loss_fn/`、`position_embedding/` |
| `src/autograd/` | 建图开关、拓扑排序和梯度累加 |
| `src/modules/`、`src/layers/` | `Module`、`Linear`、`RMSNorm`；`Embedding` 和 `SimpleDecoderLayer` |
| `src/init/` | 参数初始化：正态、均匀和 Kaiming |
| `src/models/`、`src/tokenizer/` | GPT 和 checkpoint；字节 tokenizer 和 BPE |
| `src/optim/`、`src/utils/` | SGD 和 Adam；Logger |
| `examples/`、`tests/` | 训练与生成示例；正确性、精度和 ggml 性能测试 |

### Tensor 和 Autograd

- `storage_` 持有分配地址，`data_ptr_` 指向当前 Tensor 或视图的数据起点。`shape()` 查询形状，`size(dim)` 查询单维长度，stride 的单位是元素。Tensor 拷贝共享存储，`clone()` 深拷贝。
- `to(Device)` 同设备共享存储，跨设备分配并复制。`copy_()` 写入同设备的现有存储。非连续输入的 `reshape()` 先复制成连续布局。[Tensor 设计](src/tensor/README.md)
- `Tensor::autograd_metadata_` 是 `std::shared_ptr<AutogradMetadata>`，保存 `requires_grad`、`grad_node` 和 `grad`。Tensor 拷贝共享它，视图及 `detach()` 使用新的元数据。
- `NogradNode` 控制建图，`requires_grad` 控制是否保存梯度。反向引擎用 DFS 排序，再逆序执行，临时梯度按节点指针汇总。[Autograd 设计](src/autograd/README.md)

### GPT

`train_gpt` 用 Tiny Shakespeare 训练小型 GPT，只保存最后一份 checkpoint。`chat_gpt` 加载它进行续写。两者当前使用字节 tokenizer，BPE 有独立的编解码和回归测试。[训练说明](examples/README.md)

## 构建与验证

- C++17，CMake 库目标为 `mini_infer`。公开 include 根目录是 `include/`，库内部使用 `src/`。构建产物放在 Git 忽略的 `build/`。
- CMake 检测到 CUDA Toolkit 12.4 或更新版本及可用编译器时启用 CUDA，默认目标为 SM80。用 `nvcc --version` 查看 Toolkit，用 `nvidia-smi` 查看 GPU。
- 本机用 Windows、Visual Studio 2022 和 RTX 4060 Laptop GPU 验证。按需用 `cmake/windows-cuda.cmake` 接入本机的 CUDA toolset，该文件由 Git 忽略。
- `test.sh` 构建并运行 CTest。`setup.ipynb` 在 Colab 使用远端 `main`，可运行 CUDA 测试和 benchmark。执行命令见 [tests/README.md](tests/README.md)。
