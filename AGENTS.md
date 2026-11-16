# mini_infer 项目说明（供后续 AI 会话读取）

## 目标与协作方式

- 这是为了学习现代大模型推理框架而从零手写的 C++ 项目；按实际需要实现 Tensor、CUDA、算子、模型和运行时，不追求完整对齐 PyTorch。
- 用户自己参与接口设计和实现，仍在学习 C++。解释新语法时用当前代码举小例子，说明数据所有权、调用方式和为什么这样写；用户要自己写的部分以引导和审查为主。
- 优先选容易读、容易验证的实现。不要为了抽象而新增包装层、宏、模板或文件；不要随意改已有命名、注释和接口语义。用户明确让助手实现时再直接修改。
- 中文技术说明使用 `concise-tech-docs-zh` 风格；首次使用时从当前环境的技能目录读取其 `SKILL.md`。
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

- `tensor/`：`Tensor`、`Dtype`、`Device`、FP16/BF16 编解码。`utils/`：CPU 内存分配与日志。`tests/`：CPU、CUDA Tensor、CUDA 逐元素算子、dtype 编码和 Logger 测试。
- Tensor 核心状态和构造位于 `tensor/tensor.cpp`，操作成员定义位于 `ops/tensor_ops.cpp`；CPU/CUDA 内部实现在 `ops/cpu/` 和 `ops/cuda/`。`Scalar` 位于 `tensor/scalar.h`，`fill_(const Scalar&)` 的 dtype 转换和设备分流在 `ops/tensor_ops.cpp`；逐元素反向节点位于 `autograd/backward_nodes/`。FP16/BF16 编解码在 `tensor/dtype.h`，分配/释放仍使用 `utils/memory.*`。
- 分配的存储是连续内存，Tensor 视图可以不连续；`std::shared_ptr<void> storage_` 持有分配起点并负责最终释放，`void* data_ptr_` 指向当前 Tensor 或视图的数据起点。视图共享 `storage_`，每个 Tensor 保存自己的 shape、stride、dtype 和 device。
- 拷贝构造和拷贝赋值共享数据；`clone()` 深拷贝；`copy_()` 原地复制数据。`operator[](int)` 按第一维返回共享视图，偏移按字节计算，不需要按 dtype 分支。默认空对象与零维标量的 shape 都可能为 `{}`，通过是否持有数据区分。
- 形状成员为 `shape_`，完整形状通过 `shape()` 查询（原无参 `size()` 已更名），`size(dim)` 查询单个维度。`to(Device)` 同设备返回共享句柄，跨设备分配并复制，保持 shape/dtype；`a = a.to(...)` 通过移动赋值接管结果，原存储在最后一个共享持有者释放时回收。
- `stride_` 以元素个数为单位；`narrow()` 和 `operator[]` 返回共享视图。CPU 的 `zero_`、`fill_`、`print`、`copy_` 和逐元素算子支持按 stride 访问；共享存储复制存在保守限制，非连续 `reshape()` 暂不支持。`copy_()` 要求两端 device 相同。
- CUDA 的 Tensor 构造、clone、copy_、fill_、zero_、print 和逐元素算子已实现，并按 stride 访问非连续视图；逐元素算子在 `elementwise_cuda_test` 中与 CPU 结果对照。以前的 CUDA 测试通过不代表最新改动已验证。
- Autograd 处于 WIP：已实现元数据、建图开关、反向引擎，以及逐元素、clone、to、transpose、permute 和 sum 的反向节点。DFS 后序得到前向拓扑序，再逆序处理；临时梯度表以 `const AutogradNode*` 为键，中间 Tensor 按需保存梯度，叶子直接累加持久梯度。CUDA 反向仍需目标 GPU 验证。`requires_grad` 只控制是否保存梯度，不控制建图或梯度传递。设计与任务清单见 `autograd/README.md`。
- `Logger` 默认首次写入时在进程工作目录创建 `.log`；传入文件路径可创建并追加独立日志文件。Engine 和 Worker 后续可各持有一个 `Logger` 实例。Tensor 和 CPU 分配失败只抛异常；`CUDA_CHECK` 也抛异常。`FreeCUDA` 是 `noexcept` 释放函数，失败时尝试默认日志，日志失败再写 `stderr`。

## 构建与已验证环境

- C++17，CMake 项目名 `mini_infer`，公开头文件以仓库根目录为 include 起点。CPU 测试命令见 `tests/README.md`。
- 曾在 Windows RTX 4060、CUDA Toolkit 12.6.20 和 Visual Studio 2022 环境通过 CPU/CUDA 测试；当前环境没有完整 CUDA Toolkit，仅有供编辑器补全的头文件，不能据此宣称当前版本已通过 GPU 验证。构建产物在被忽略的 `build/`。
- `.vscode/` 使用 CMake Tools 提供 C/C++ 的编译器、include 和宏定义，不保存机器绝对路径；额外将被忽略的 `build/cuda-headers-12.4/include` 作为本地编辑补全的可选路径，不影响构建。本机在被忽略的 `cmake/windows-cuda.cmake` 中通过 `CUDA_PATH` 获取 Toolkit 自带的 VS 集成文件；该文件存在时才自动加载，标准构建不依赖此本地补丁。
- Colab Pro 曾连接 NVIDIA L4（SM89）；历史构建结果不能代替当前代码的 GPU 验证。
- `setup.ipynb` 依次检查 GPU、从 GitHub 同步源码、构建 CUDA、单独运行 CTest；它拉取的是**已推送版本**，不会自动包含本地未提交改动。`test.sh` 可在 Linux/Colab 或 Windows Git Bash 下执行 CUDA 构建与 CTest。
- CUDA 目标：Ampere 及更新架构（SM80 起）；CMake 自动检测 Toolkit >= 12.4 和可用 CUDA 编译器，成功时定义 `ENABLE_CUDA`，否则只构建 CPU。仅有 Toolkit 而无可用 GPU/驱动时，CUDA 测试跳过。默认编译 SM80 并保留 PTX，支持更新 GPU。`nvidia-smi` 的 CUDA 版本不等于已安装 Toolkit 版本，应以 `nvcc --version` 为准。

## 下一步：Autograd（训练支持）

1. 按 `autograd/README.md` 的任务清单推进。逐元素、clone、to、transpose、permute 和 sum 已接入反向；CUDA 新代码需在目标 GPU 验证，其他视图的反向仍待实现。
2. Tensor 持有共享的 `AutogradMetadata`（`requires_grad`、`grad_node`、`grad`）；拷贝共享 meta，视图新建 meta。Node 只保存输入、不保存输出；引擎拓扑序保存输出 Tensor，通过 `grad_node()` 取得 Node，在 `NogradNode` 下逆序调用，临时表以 Node 指针为键，叶子直接累加持久梯度。
3. 第一阶段以单元素 loss 和浮点梯度为主；`requires_grad` 只控制是否保存该 Tensor 的梯度，与 PyTorch 命名语义不同，不控制建图或梯度传递。`operator[]`、`narrow` 和 `reshape` 的反向，以及其他归约与矩阵乘仍待补齐。
4. 用户实现核心代码，助手负责测试、运行验证和审查；每完成一项并通过测试再进入下一项。
5. 完成后在目标 GPU 环境用 `test.sh` 重跑回归；历史 GPU 结果不替代对最新版本的验证。

## 其他环境说明

- Colab MCP 曾安装但浏览器连接未成功；当前开发流程是本地 VS Code 写代码、本地 GPU 编译与测试，Colab 用于补充验证，MCP 不是前置条件。用户已明确不想再为 MCP 连接反复排查。
