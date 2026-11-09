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

- **一个功能只对应一条提交。** 功能尚未完成时，在类型前缀后标注 `WIP:`，例如 `[refact] WIP: 拆分 Tensor 后端`，准确写明正在开发的功能；不要把同一功能的实现、修复、测试、文档或构建脚本拆成多条提交。用户当次明确指定的提交格式和范围优先。
- 功能开发期间，后续所有属于该功能的修改都暂存到这条 WIP 提交，使用 `git commit --amend --no-edit` 更新，再使用 `git push --force-with-lease` 更新远端。不得为同一功能新建提交，也不得用普通 `git push` 留下多条 WIP 提交。
- 只有该功能实现完成，并通过相应测试及目标环境验证后，才在**最后一次 amend** 时移除标题中的 `WIP:`，保留 `[类型]` 前缀并准确描述已完成功能，然后推送。功能未完成或验证未通过时，始终保留 `WIP:`；用户明确要求提交当前快照时，在正文如实说明未完成项。
- 开始新的独立功能时，才创建新的 WIP 提交。每次 amend 前核对 `HEAD` 确实是该功能的提交，并检查待暂存文件；不得把其他功能或用户未授权的本地改动混入。
- 如果拿不准功能边界、应 amend 哪条提交、提交信息该怎么写，或功能是否达到完成条件，**先问用户**，得到明确答复后再提交或推送；不要自行猜测。
- 改写已推送的提交时只用 `--force-with-lease`，先核对远端状态；不要使用无保护的 `--force`。用户当次明确要求的提交方式优先。

## C/C++ include 顺序

- 适用于项目中的 C/C++ 头文件和源文件，包括后续的 `.cu` / `.cuh`。
- 先集中放系统、标准库和第三方头文件，使用 `#include <...>`；然后空一行，集中放项目自己的头文件，使用 `#include "..."`。两个组内不额外空行。
- 项目头文件按实际需要和依赖顺序排列；`.cpp` 等实现文件对应的头文件放在项目头文件组的最前面。不要把它单独放到系统头文件前面。
- `#pragma once` 保持在头文件开头，与 include 区域空一行。整理顺序时不要改动接口或实现逻辑。

## 当前设计

- `tensor/`：`Tensor`、`Dtype`、`Device`、FP16/BF16 编解码。`utils/`：CPU 内存分配与日志。`tests/`：CPU、CUDA Tensor、CUDA 逐元素算子、dtype 编码和 Logger 测试。
- Tensor 实现分为 `tensor.cpp`（公共逻辑与设备分流）、`tensor_cpu.cpp`（CPU 复制、填充、清零及打印）和 `tensor_cuda.cu`（CUDA 复制、清零及 fill kernel）。`fill_<T>()` 在头文件中转换输入，再调用私有 `FillValue()` 分流。FP16/BF16 编解码统一放在 `dtype.h`，分配/释放继续使用 `utils/memory.*`。
- 分配的存储是连续内存，Tensor 视图可以不连续；`std::shared_ptr<void> storage_` 持有分配起点并负责最终释放，`void* data_ptr_` 指向当前 Tensor 或视图的数据起点。视图共享 `storage_`，每个 Tensor 保存自己的 shape、stride、dtype 和 device。
- 拷贝构造和拷贝赋值共享数据；`clone()` 深拷贝；`copy_()` 原地复制数据。`operator[](int)` 按第一维返回共享视图，偏移按字节计算，不需要按 dtype 分支。默认空对象与零维标量的 shape 都可能为 `{}`，通过是否持有数据区分。
- 形状成员为 `shape_`，完整形状通过 `shape()` 查询（原无参 `size()` 已更名），`size(dim)` 查询单个维度。`to(Device)` 同设备返回共享句柄，跨设备分配并复制，保持 shape/dtype；`a = a.to(...)` 通过移动赋值接管结果，原存储在最后一个共享持有者释放时回收。
- `stride_` 以元素个数为单位；`narrow()` 和 `operator[]` 返回共享视图。CPU 的 `zero_`、`fill_`、`print`、`copy_` 和逐元素算子已支持按 stride 访问；共享存储复制存在保守限制，非连续 `clone()`、跨设备 `to()` 和 `reshape()` 暂不支持。`copy_()` 要求两端 device 相同。
- CUDA 的 Tensor 构造、clone、copy_、fill_、zero_、print 和逐元素算子已实现，并按 stride 访问非连续视图；逐元素算子在 `elementwise_cuda_test` 中与 CPU 结果对照。以前的 CUDA 测试通过不代表最新改动已验证。
- `Logger` 默认首次写入时在进程工作目录创建 `.log`；传入文件路径可创建并追加独立日志文件。Engine 和 Worker 后续可各持有一个 `Logger` 实例。Tensor 和 CPU 分配失败只抛异常；`CUDA_CHECK` 也抛异常。`FreeCUDA` 是 `noexcept` 释放函数，失败时尝试默认日志，日志失败再写 `stderr`。

## 构建与已验证环境

- C++17，CMake 项目名 `mini_infer`，公开头文件以仓库根目录为 include 起点。CPU 测试命令见 `tests/README.md`。
- 曾在 Windows RTX 4060、CUDA Toolkit 12.6.20 和 Visual Studio 2022 环境通过 CPU/CUDA 测试；当前环境没有完整 CUDA Toolkit，仅有供编辑器补全的头文件，不能据此宣称当前版本已通过 GPU 验证。构建产物在被忽略的 `build/`。
- `.vscode/` 使用 CMake Tools 提供 C/C++ 的编译器、include 和宏定义，不保存机器绝对路径，也不再依赖临时 CUDA 头文件。本机在被忽略的 `cmake/windows-cuda.cmake` 中通过 `CUDA_PATH` 获取 Toolkit 自带的 VS 集成文件；该文件存在时才自动加载，标准构建不依赖此本地补丁。
- Colab Pro 已连接过 NVIDIA L4（SM89）。`nvidia-smi` 显示驱动支持 CUDA 13.0；`nvcc --version` 实测 Toolkit 13.0.88。`setup.ipynb` 保存的 Colab CUDA 构建和 4 项 CTest 均通过，其中包括 `tensor_cuda_test`。
- `setup.ipynb` 依次检查 GPU、从 GitHub 同步源码、构建 CUDA、单独运行 CTest；它拉取的是**已推送版本**，不会自动包含本地未提交改动。`test.sh` 可在 Linux/Colab 或 Windows Git Bash 下执行 CUDA 构建与 CTest。
- CUDA 目标：Ampere 及更新架构（SM80 起）；CMake 自动检测 Toolkit >= 12.4 和可用 CUDA 编译器，成功时定义 `ENABLE_CUDA`，否则只构建 CPU。仅有 Toolkit 而无可用 GPU/驱动时，CUDA 测试跳过。默认编译 SM80 并保留 PTX，支持更新 GPU。`nvidia-smi` 的 CUDA 版本不等于已安装 Toolkit 版本，应以 `nvcc --version` 为准。

## 下一步：CUDA stride 与非连续操作

1. 优先设计并实现 CUDA Tensor 非连续视图操作，从一开始考虑 shape、以元素为单位的 stride、当前数据起点、设备端元数据及所有权；先讨论接口与元数据传递方式，再逐项实现 `zero_`、`fill_`、`copy_`、`print`。连续路径可按需保留优化，但不能假定输入连续。
2. 完善非连续 `clone()` 和跨设备 `to()` 的布局及复制语义；明确共享存储、源目标重叠时的处理。当前 `copy_()` 只支持同设备，扩展跨设备能力前先讨论接口。
3. 再实现 `ops/` 的 CUDA 逐元素算子，设计时直接支持输入 stride 和连续输出；处理 dtype（包括 FP16/BF16）、线程越界、错误检查及与 CPU 一致的数值语义。用户先参与实现一个算子，其余再按实际要求协作。
4. 在具备 CUDA Toolkit 和 GPU 的目标环境编译并测试：覆盖连续和非连续输入、不同 stride、嵌套视图、标量、dtype、线程块边界和复制限制；CPU 回归也需通过。已有的历史 GPU 结果不替代对最新版本的验证。
5. 在功能正确并经过实际测量后，再考虑连续快速路径或 CUDA kernel 优化。不得先仅实现连续 CUDA 算子再推迟 stride 布局设计。

## 其他环境说明

- Colab MCP 曾安装但浏览器连接未成功；当前开发流程是本地 VS Code 写代码、本地 GPU 编译与测试，Colab 用于补充验证，MCP 不是前置条件。用户已明确不想再为 MCP 连接反复排查。
