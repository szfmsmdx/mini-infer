# Tensor 测试

需要支持 C++17 的编译器和 CMake 3.18+。从仓库根目录运行：

```powershell
cmake -S . -B build/tests -DBUILD_TESTING=ON
cmake --build build/tests --config Release --parallel 2
ctest --test-dir build/tests -C Release --output-on-failure
```

修改代码后，重新执行后两条命令。若提示找不到命令，检查 PATH，或重启终端及启动它的应用。

| 测试程序 | 覆盖内容 |
| --- | --- |
| `logger_test` | Engine/Worker 独立文件、日志级别、追加写入及无效路径 |
| `tensor_cpu_test` | 构造、shape 查询、空对象与标量、共享拷贝、移动、索引视图与生命周期、clone、copy_、同设备 to、禁用 CUDA 时的迁移异常、填充、清零、打印和解码 |
| `dtype_encode_test` | FP16/BF16 编码、舍入、特殊值、所有非 NaN 编码往返，以及 16 位 Tensor 填充 |
| `tensor_cuda_test` | CUDA 内存、构造、索引视图、所有权、clone、copy_、8 种 dtype 双向 to 与原始编码保留、迁移赋值和共享生命周期、标量/空对象迁移、填充、清零、打印及异常 |

## `.to()` 覆盖范围

- CPU/CUDA 同设备共享、同设备自赋值，保留索引视图的偏移和存储生命周期。
- 8 种 dtype 的 CPU→CUDA、CUDA→CPU 分别验证，以及完整往返，保持 shape、dtype 和数据。
- 跨设备结果独立于源存储；源对象销毁后，迁移结果及同设备共享视图仍然有效。
- `a = a.to(...)` 双向赋值后，原设备上的其他 Tensor/视图继续有效；迁移失败时原对象的数据指针、元数据和值保持不变。
- 零维标量、三维 Tensor 的多级索引视图、默认空对象，以及移动后保留非默认 dtype 的空对象。
- FP16/BF16 原始编码、Float32 负零/无穷/NaN payload，以及超出浮点精确表示范围的 Int64 值按字节保留。
- 非法目标设备和未启用 CUDA 时的报错；禁用 CUDA 后空对象也不能迁移到 CUDA。

尚未注入 CPU/CUDA 分配失败或 CUDA 复制运行时错误，因此这些失败路径的资源回收与异常安全未得到故障注入验证。最后一个共享持有者释放时是否存在泄漏，也尚未使用专门的内存检测工具验证。上述错误路径需要独立的故障注入或内存检测环境，不通过耗尽本机内存/显存来测试。

测试使用 `Check()` 抛异常，Release 构建也会执行检查；CTest 的 3/4 项数量是测试程序数量，`.to()` 用例包含在两个 Tensor 测试程序内，不代表只有 3/4 个检查。

只跑其中一组，例如 Tensor：

```powershell
ctest --test-dir build/tests -C Release -R "^tensor_cpu_test$" --output-on-failure
```

CMake 自动检测 Toolkit >= 12.4 和可用 CUDA 编译器，满足条件时编译 CUDA，否则只构建 CPU。`ENABLE_CUDA` 是自动检测结果，无需手动设置。更换工具链后使用新的构建目录重新检测，避免复用旧缓存。
仅有 Toolkit 而无可用 GPU/驱动时，CUDA 测试显示为跳过。
默认 Logger 的 `.log` 位于程序运行时的工作目录；CTest 运行时该目录是构建目录。

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

上面的 `89` 对应 Colab L4 和 RTX 4060；其他 GPU 应改成其对应架构。`setup.ipynb` 保存的 Colab L4 运行结果显示 4/4 项测试通过。

VS Code 安装 C/C++ 与 CMake Tools 后运行 `CMake: Configure`，由 CMake Tools 提供实际编译器、include 和宏定义；`.vscode/` 不保存机器绝对路径，不再依赖临时 CUDA 头文件。首次配置如需选择 Kit，Windows CUDA 开发选择 Visual Studio x64。
