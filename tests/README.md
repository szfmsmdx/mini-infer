# 测试

正确性测试在 `tests/CMakeLists.txt` 注册到 CTest。与 ggml 对比的性能程序由 `tests/benchmarks/CMakeLists.txt` 按需构建。

需要支持 C++17 的编译器和 CMake 3.18+。从仓库根目录运行：

```powershell
cmake -S . -B build/tests -DBUILD_TESTING=ON
cmake --build build/tests --config Release --parallel 2
ctest --test-dir build/tests -C Release --output-on-failure
```

修改代码后，重新执行后两条命令。若提示找不到命令，检查 PATH，或重启终端及启动它的应用。

| 测试程序 | 覆盖内容 |
| --- | --- |
| `reduction_cpu_test` | sum 的 8 种 dtype、整数提升与回绕、非连续视图和端到端反向 |
| `reduction_cuda_test` | sum 的 8 种 dtype、线程数边界、跨步循环、非连续视图和端到端反向 |
| `layout_cpu_test` | `permute` 的维度排列、共享存储、非法输入、反向梯度位置和端到端传播 |
| `scalar_test` | 标量转换与构造、零维判断、标量 Tensor 填充、full、共享存储及 SumNode 直接反向 |
| `autograd_cpu_test` | 梯度开关、元数据共享、直接反向公式、保护路径；端到端反向、共享分支、重复累加、三种浮点 dtype 与有限差分 |
| `logger_test` | Engine/Worker 独立文件、日志级别、追加写入及无效路径 |
| `tensor_cpu_test` | 构造、shape 查询、空对象与标量、共享拷贝、移动、索引视图与生命周期、clone、copy_、同设备 to、禁用 CUDA 时的迁移异常、填充、清零、打印和解码 |
| `elementwise_cpu_test` | `add()` / `sub()` / `mul()` / `div()` / `neg()` 的 Float32、整数回绕与除零边界、FP16/BF16、无穷与 NaN、零维标量与索引视图、输入不变、输出独立，以及空对象、shape、dtype 错误 |
| `dtype_encode_test` | FP16/BF16 编码、舍入、特殊值、所有非 NaN 编码往返，以及 16 位 Tensor 填充 |
| `autograd_cuda_test` | 跨设备双向及链式反向、建图开关、整数 dtype、非连续输入的节点连接 |
| `tensor_cuda_metadata_test` | CUDA 构造、迁移、clone、索引、narrow 和 reshape 的 shape/stride 设备副本；CPU、空对象与标量；拷贝、移动及共享生命周期 |
| `tensor_cuda_test` | CUDA 内存、构造、索引视图、所有权、clone、copy_、8 种 dtype 双向 to 与原始编码保留、迁移赋值和共享生命周期、标量/空对象迁移、填充、清零、打印及异常 |
| `elementwise_cuda_test` | 8 种 dtype 的 add/sub/mul/div/neg 与 CPU 结果逐字节对照；覆盖 stride 视图、标量、线程块边界、整数回绕与除零、FP16/BF16 特殊值及错误输入 |

## 测试进度

- [x] CPU Tensor、逐元素算子、归约及 Autograd 测试。
- [x] CUDA Tensor、逐元素算子、设备迁移和归约的测试代码。
- [x] 在 Colab A100 运行 CPU/CUDA 测试，CTest 13/13 通过。
- [ ] 补充设备分配和复制失败的故障注入、内存泄漏检查。

## `.to()` 覆盖范围

- CPU/CUDA 同设备共享、同设备自赋值，保留索引视图的偏移和存储生命周期。
- 8 种 dtype 的 CPU→CUDA、CUDA→CPU 分别验证，以及完整往返，保持 shape、dtype 和数据。
- 跨设备结果独立于源存储；源对象销毁后，迁移结果及同设备共享视图仍然有效。
- `a = a.to(...)` 双向赋值后，原设备上的其他 Tensor/视图继续有效；迁移失败时原对象的数据指针、元数据和值保持不变。
- 零维标量、三维 Tensor 的多级索引视图、默认空对象，以及移动后保留非默认 dtype 的空对象。
- FP16/BF16 原始编码、Float32 负零/无穷/NaN payload，以及超出浮点精确表示范围的 Int64 值按字节保留。
- 非法目标设备和未启用 CUDA 时的报错；禁用 CUDA 后空对象也不能迁移到 CUDA。

测试使用 `Check()` 抛异常，Release 构建也会执行检查。

`autograd_cpu_test` / `autograd_cuda_test` 新增梯度管理与版本检测用例：清梯度保留存储、再次反向、别名/视图/detach 的版本共享、fill_/zero_/copy_ 的版本递增、mul/div 修改输入后的报错、独立副本不误报，以及反向后更新参数再重新前向的流程。覆盖 Float32、FP16、BF16，本地 CPU 回归 8/8 通过，用户在 Colab 确认 CPU/CUDA 回归全部通过。

`reduction_cpu_test` / `reduction_cuda_test` 还覆盖二元逐元素算子的自动广播：

- `add/sub/mul/div` 的双边广播、前向数值和两端梯度 shape 与数值。
- 标量与非连续视图的双向运算、关闭建图及不兼容 shape。
- Float32、FP16、BF16、Int32 的多前导维广播；浮点梯度归约及整数不建图。
- 同一叶子经不同广播路径参与计算时的梯度累加。

最新自动广播用例已在本地通过 CPU 回归（8/8）；用户在 Colab 完成包含新增用例的 CPU/CUDA 回归，CTest 13/13 通过。

只跑其中一组，例如 Tensor：

```powershell
ctest --test-dir build/tests -C Release -R "^tensor_cpu_test$" --output-on-failure
```

CMake 自动检测 Toolkit >= 12.4 和可用 CUDA 编译器，满足条件时编译 CUDA，否则只构建 CPU。`ENABLE_CUDA` 是自动检测结果，无需手动设置。更换工具链后使用新的构建目录重新检测，避免复用旧缓存。
仅有 Toolkit 而无可用 GPU/驱动时，CUDA 测试显示为跳过。
默认 Logger 的 `.log` 位于程序运行时的工作目录。CTest 默认从构建目录下的 `tests/` 运行正确性测试。

在有 CUDA Toolkit 和 GPU 的 Colab 环境，从已同步最新源码的仓库根目录运行：

```bash
bash ./test.sh
```

Windows + RTX 4060 可在 VS Code 的 Git Bash 集成终端运行同一命令。脚本默认使用 CUDA 架构 `80` 并保留 PTX，适用于 Ampere 及更新 GPU。CMake 从当前环境选择编译器和生成器；需要指定生成器时可设置 `CMAKE_GENERATOR`，不要将机器上的编译器路径写进仓库。CUDA 编译需要 Toolkit 和受支持的 MSVC C++ 工具链；找不到 CUDA 编译环境时，项目只构建 CPU。本机额外使用被忽略的 `cmake/windows-cuda.cmake` 从 `CUDA_PATH` 读取 VS 集成文件，补齐未注册的 CUDA toolset；该补丁存在时才加载，标准安装环境无需此文件。

其他 GPU 可在运行前指定架构，例如 `CUDA_ARCH=80 bash ./test.sh`。也可以手动执行下面的 CMake/CTest 命令：

```bash
cmake -S . -B build/tests -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/tests --config Release --parallel 2
ctest --test-dir build/tests -C Release --output-on-failure
```

上面的 `89` 对应 Colab L4 和 RTX 4060；其他 GPU 应改成其对应架构。

VS Code 安装 C/C++ 与 CMake Tools 后运行 `CMake: Configure`，由 CMake Tools 提供实际编译器、include 和宏定义；`.vscode/` 不保存机器绝对路径。未安装完整 Toolkit 时，可选的 `build/cuda-headers-12.4/include` 仅供本地补全，不参与 CUDA 构建。首次配置如需选择 Kit，Windows CUDA 开发选择 Visual Studio x64。

本轮视图与布局扩展（reshape/narrow/索引反向、contiguous/squeeze/unsqueeze、reshape 推导与非连续输入、公共检查重构）已由用户在 Colab 完成 CPU/CUDA 回归，CTest 13/13 通过，总耗时 2.30 秒；本地 CPU 回归 8/8 通过。

## Matmul 正确性与 CPU 性能对比

`matmul_cpu_test` 和 `matmul_cuda_test` 通过 CTest 验证二维 matmul，覆盖 8 种 dtype、非连续输入、低精度累加和整数回绕。CUDA 用例还覆盖输出元素数超过一个 block 的线程数。用户在 Colab 完成 CPU/CUDA 回归，CTest 15/15 通过。

性能测试单独构建，唯一对照是 llama.cpp 的 ggml CPU 后端。

```sh
cmake -S . -B build/bench -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/bench --config Release --target matmul_cpu_bench
```

运行 `build/bench/tests/benchmarks/matmul_cpu_bench`。Visual Studio 生成器的可执行文件在该目录下的 `Release/` 中。

### 依赖版本

`3rdparty/CMakeLists.txt` 中的 `LLAMA_CPP_REVISION` 是 llama.cpp 的版本锁定。开启 benchmark 时，CMake 从该 commit 下载源码归档并构建其中的 ggml。已有同版本源码时，可传入 `-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<目录>`。更新版本时，在同一次变更里更新 commit、验证构建并重跑性能对比。

### 用例尺寸

当前程序运行快速组。FP32，固定 `K=N=1024`，取 `M=1/32/128`，分别测行主序 B 和转置权重。该组用于检查 benchmark 能运行和输出一致。

模型层尺寸组用于评估 FP32 kernel 优化，目前已在 CUDA benchmark 提供。形状按 `A[M,K] × B[K,N]` 记录，保持两侧各自的 dtype 和执行配置后再比较不同 M。

| M | K | N | 覆盖场景 |
|---:|---:|---:|---|
| 1、2、4、8 | 4096 | 4096 | 单行 decode 到小批 decode |
| 32、128、512 | 4096 | 4096 | 多行输入，观察权重复用随 M 的变化 |
| 1、32、512 | 4096 | 14336 | FFN 上投影 |
| 1、32、512 | 14336 | 4096 | FFN 下投影 |

模型层尺寸组以转置权重为主。评估行主序快路径或 tiling 时，另测同形状的行主序 B，并单独报告结果。

### 计时与报告

mini-infer 当前串行计算。ggml 使用算子 API 的上游默认线程配置 `GGML_DEFAULT_N_THREADS`，锁定版本为 4。基准输出会打印两侧的线程配置。

每组先预热 3 次并校验输出，再交替测量两侧各 10 次。报告均值、中位数和按均值计算的 GFLOP/s。预分配和建图在计时外。行主序 B 需要 ggml 在图中转置并连续化，转换计入耗时。转置权重可直接传给 ggml。两种布局的结果分别解释。

性能报告记录两侧代码版本、CPU、编译器和优化选项、线程数、dtype、M/K/N、布局及计时范围。共享 CI 跑正确性和 benchmark 构建检查。需要设置性能门槛时，在固定机器上建立测量波动范围后再启用。

## CUDA matmul 性能对比

在 Colab GPU runtime 按顺序运行 `setup.ipynb`。CMake 单元格根据 GPU 计算能力选择 CUDA 架构，在 `build/cuda-colab` 一次构建 Release 版测试和 benchmark。接下来的 CTest 和 CUDA perf 单元格分别运行已构建的程序。

默认测 `M=1/32/128, K=N=1024`。在最后一个单元格的命令末尾加 `--model`，可运行表中的 13 个模型层尺寸。当前单 block 路径跑大尺寸耗时较长。

两侧在同一张 GPU 上使用相同的 FP32 输入和 `weight.T` 布局。ggml 运行单个 `ggml_mul_mat` 节点，采用 llama.cpp 的默认 CUDA Graph 配置和 kernel 分发。预分配、上传和输出核对在计时外。预热 3 次后，两侧交替测量各 10 次，计时包含提交和等待 GPU 完成，输出均值、中位数、GFLOP/s 和耗时比。
