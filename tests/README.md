# Tensor 测试

需要支持 C++17 的 GCC、CMake 3.20+ 和 Ninja；本机 WinLibs 工具链已验证通过。从仓库根目录运行：

```powershell
cmake -S . -B build/cpu-tests -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
cmake --build build/cpu-tests
ctest --test-dir build/cpu-tests --output-on-failure
```

修改代码后，重新执行后两条命令。若提示找不到命令，检查 PATH，或重启终端及启动它的应用。

| 测试程序 | 覆盖内容 |
| --- | --- |
| `logger_test` | Engine/Worker 独立文件、日志级别、追加写入及无效路径 |
| `tensor_cpu_test` | 构造、空对象与标量、共享拷贝、移动、索引视图与生命周期、clone、copy_、填充、清零、异常、打印和解码 |
| `dtype_encode_test` | FP16/BF16 编码、舍入、特殊值、所有非 NaN 编码往返，以及 16 位 Tensor 填充 |
| `tensor_cuda_test` | CUDA 内存、构造、索引视图、所有权、clone、copy_、8 种 dtype 填充和清零、打印及异常 |

只跑其中一组，例如 Tensor：

```powershell
ctest --test-dir build/cpu-tests -R "^tensor_cpu_test$" --output-on-failure
```

默认 `MINI_INFER_ENABLE_CUDA=OFF`，本机没有 CUDA Toolkit 或 GPU 时只编译并运行 CPU 和 Logger 测试。
默认 Logger 的 `.log` 位于程序运行时的工作目录；CTest 运行时该目录是 `build/cpu-tests`。

在有 CUDA Toolkit 和 GPU 的 Colab 环境，从已同步最新源码的仓库根目录运行：

```bash
bash ./test.sh
```

Windows + RTX 4060 可在 VS Code 的 Git Bash 集成终端运行同一命令。脚本默认使用 CUDA 架构 `89`；Windows 默认选择 Visual Studio 2022 生成器。需要安装 CUDA Toolkit 和受支持的 MSVC C++ 工具链；仅有 Git Bash 或 WinLibs GCC 不足以编译 Windows CUDA 代码。

其他 GPU 可在运行前指定架构，例如 `CUDA_ARCH=80 bash ./test.sh`。也可以手动执行下面的 CMake/CTest 命令：

```bash
cmake -S . -B build/cuda-tests -DMINI_INFER_ENABLE_CUDA=ON -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda-tests -j2
ctest --test-dir build/cuda-tests --output-on-failure
```

上面的 `89` 对应 Colab L4 和 RTX 4060；其他 GPU 应改成其对应架构。`setup.ipynb` 保存的 Colab L4 运行结果显示 4/4 项测试通过。
