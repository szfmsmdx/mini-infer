# mini_infer 项目说明（供后续 AI 会话读取）

## 目标与协作方式

- 这是为了学习现代大模型推理框架而从零手写的 C++ 项目；按实际需要实现 Tensor、CUDA、算子、模型和运行时，不追求完整对齐 PyTorch。
- 用户自己参与接口设计和实现，仍在学习 C++。解释新语法时用当前代码举小例子，说明数据所有权、调用方式和为什么这样写；用户要自己写的部分以引导和审查为主。
- 优先选容易读、容易验证的实现。不要为了抽象而新增包装层、宏、模板或文件；不要随意改已有命名、注释和接口语义。用户明确让助手实现时再直接修改。
- 中文技术说明使用 `concise-tech-docs-zh` 风格。首次使用时读取 `C:/Users/Cloudsway/.codex/skills/concise-tech-docs-zh/SKILL.md`。
- 新会话开始先看 `git status --short` 和相关源码。区分“本地工作区已修改”和“已推送到 GitHub”，不要覆盖用户的未提交改动；提交或推送时遵守用户当次要求。

## Git 提交规则（必须遵守）

- **一个功能只对应一条提交。** 功能尚未完成时，提交信息必须以 `WIP:` 开头，准确写明正在开发的功能；不要把同一功能的实现、修复、测试、文档或构建脚本拆成多条提交。
- 功能开发期间，后续所有属于该功能的修改都暂存到这条 WIP 提交，使用 `git commit --amend --no-edit` 更新，再使用 `git push --force-with-lease` 更新远端。不得为同一功能新建提交，也不得用普通 `git push` 留下多条 WIP 提交。
- 只有该功能实现完成，并通过相应测试及目标环境验证后，才在**最后一次 amend** 时将 `WIP:` 提交信息改为准确描述已完成功能的正式信息，然后推送。功能未完成或验证未通过时，始终保留 `WIP:`。
- 开始新的独立功能时，才创建新的 WIP 提交。每次 amend 前核对 `HEAD` 确实是该功能的提交，并检查待暂存文件；不得把其他功能或用户未授权的本地改动混入。
- 如果拿不准功能边界、应 amend 哪条提交、提交信息该怎么写，或功能是否达到完成条件，**先问用户**，得到明确答复后再提交或推送；不要自行猜测。
- 改写已推送的提交时只用 `--force-with-lease`，先核对远端状态；不要使用无保护的 `--force`。用户当次明确要求的提交方式优先。

## C/C++ include 顺序

- 适用于项目中的 C/C++ 头文件和源文件，包括后续的 `.cu` / `.cuh`。
- 先集中放系统、标准库和第三方头文件，使用 `#include <...>`；然后空一行，集中放项目自己的头文件，使用 `#include "..."`。两个组内不额外空行。
- 项目头文件按实际需要和依赖顺序排列；`.cpp` 等实现文件对应的头文件放在项目头文件组的最前面。不要把它单独放到系统头文件前面。
- `#pragma once` 保持在头文件开头，与 include 区域空一行。整理顺序时不要改动接口或实现逻辑。

## 当前设计

- `tensor/`：`Tensor`、`Dtype`、`Device`、FP16/BF16 编解码。`utils/`：CPU 内存分配与日志。`tests/`：CPU、CUDA Tensor、dtype 编码和 Logger 测试。
- 连续内存；`std::shared_ptr<void> storage_` 持有分配起点并负责最终释放，`void* data_ptr_` 指向当前 Tensor 或索引视图的数据起点。视图共享 `storage_`，形状、dtype、device 由每个 Tensor 自己保存。
- 拷贝构造和拷贝赋值共享数据；`clone()` 深拷贝；`copy_()` 原地复制数据。`operator[](int)` 按第一维返回共享视图，偏移按字节计算，不需要按 dtype 分支。默认空对象与零维标量的 shape 都可能为 `{}`，通过是否持有数据区分。
- CPU 的构造、拷贝/移动、索引、clone、copy_、fill_、zero_、print 和元数据查询已实现；FP16/BF16 编码、舍入及打印解码已测试。当前 `copy_()` 要求两端 device 相同，不支持跨设备复制。
- CUDA 构造、clone、copy_、fill_、zero_、print，以及 CUDA 分配/释放已写入源码；Colab L4 已编译运行，4 项 CTest 通过。CPU 和 CUDA 测试按 CMake 选项分开编译。
- `Logger` 默认首次写入时在进程工作目录创建 `.log`；传入文件路径可创建并追加独立日志文件。Engine 和 Worker 后续可各持有一个 `Logger` 实例。Tensor 和 CPU 分配失败只抛异常；`CUDA_CHECK` 也抛异常。`FreeCUDA` 是 `noexcept` 释放函数，失败时尝试默认日志，日志失败再写 `stderr`。

## 构建与已验证环境

- C++17，CMake 项目名 `mini_infer`，公开头文件以仓库根目录为 include 起点。CPU 测试命令见 `tests/README.md`。
- 本地 Windows：WinLibs GCC、CMake、Ninja；`build/cpu-tests` 中三组 CTest 已通过。构建产物在 `build/`，被 `.gitignore` 忽略。
- 本地没有完整 CUDA Toolkit 或 `nvcc`。VS Code 已安装 C/C++、clangd、Nsight 和 Colab 扩展；`.vscode/` 配置了 `.cu/.cuh` 文件关联和代码提示路径。CUDA 12.4 开发头文件临时放在忽略的 `build/cuda-headers-12.4/include`，不能把它当作可在本机编译 CUDA 的工具链。
- Colab Pro 已连接过 NVIDIA L4（SM89）。`nvidia-smi` 显示驱动支持 CUDA 13.0；`nvcc --version` 实测 Toolkit 13.0.88。`setup.ipynb` 保存的 Colab CUDA 构建和 4 项 CTest 均通过，其中包括 `tensor_cuda_test`。
- `setup.ipynb` 依次检查 GPU、从 GitHub 同步源码、构建 CUDA、单独运行 CTest；它拉取的是**已推送版本**，不会自动包含本地未提交改动。`test.sh` 可在 Linux/Colab 或 Windows Git Bash 下执行 CUDA 构建与 CTest。
- CUDA 目标：Ampere 及更新架构（SM80 起）；CMake 的 `MINI_INFER_ENABLE_CUDA` 默认为 OFF，开启时要求 CUDA Toolkit >= 12.4。`nvidia-smi` 的 CUDA 版本不等于已安装 Toolkit 版本，应以 `nvcc --version` 为准。

## 下一步：CUDA Tensor

1. 在 Windows RTX 4060 笔记本安装 CUDA Toolkit 与受支持的 MSVC 后，用 Git Bash 运行 `bash ./test.sh` 验证 GPU 构建和测试。
2. 按实际测量结果再考虑 CUDA fill kernel 优化；跨设备 `copy_()` 等接口语义尚未定，不要擅自改动。

## 其他环境说明

- Colab MCP 曾安装但浏览器连接未成功；当前开发流程是本地 VS Code 写代码，Colab 承担云端编译与测试，MCP 不是前置条件。用户已明确不想再为 MCP 连接反复排查。
