# CPU Tensor 测试

需要支持 C++17 的 GCC、CMake 3.20+ 和 Ninja；本机 WinLibs 工具链已验证通过。从仓库根目录运行：

```powershell
cmake -S . -B build/cpu-tests -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
cmake --build build/cpu-tests
ctest --test-dir build/cpu-tests --output-on-failure
```

修改代码后，重新执行后两条命令。若提示找不到命令，检查 PATH，或重启终端及启动它的应用。

| 测试程序 | 覆盖内容 |
| --- | --- |
| `tensor_cpu_test` | 构造、空对象与标量、共享拷贝、移动、索引视图与生命周期、clone、copy_、填充、清零、异常、打印和解码 |
| `dtype_encode_test` | FP16/BF16 编码、舍入、特殊值、所有非 NaN 编码往返，以及 16 位 Tensor 填充 |

只跑其中一组，例如 Tensor：

```powershell
ctest --test-dir build/cpu-tests -R "^tensor_cpu_test$" --output-on-failure
```

CUDA 尚未实现；测试覆盖 CPU 功能，以及 CUDA 构造占位分支的异常，不依赖 GPU。
