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
| `sqrt_cpu_test` / `sqrt_cuda_test` | sqrt、rsqrt 的 8 种输入 dtype、整数提升、跨步视图、特殊值、浮点梯度和原地修改检测；rsqrt 低精度反向失败样本、不同 stride 的输入与上游梯度、跨 block 长度和零维标量 |
| `exp_cpu_test` / `exp_cuda_test` | exp 的 8 种输入 dtype、转置视图、跨 block 长度，以及复用输出的反向和版本检查 |
| `activation_cpu_test` / `activation_cuda_test` | sigmoid 和 SiLU 的浮点前向、梯度和转置输入；sigmoid 极端梯度的 FP32 中间计算、不同 stride 的输入与上游梯度、零维标量；SiLU 负尾部、计算分支边界、特殊值、no-grad 和输入版本检查；5 种整数输入提升及 sigmoid 输出版本检查 |
| `rmsnorm_cpu_test` / `rmsnorm_cuda_test` | Module 前向、输入/权重梯度、梯度累加、no-grad、错误形状和原地修改检测 |
| `rmsnorm_precision_cpu_test` / `rmsnorm_precision_cuda_test` | 三种浮点 dtype、10 个固定案例的前向与两个梯度，数学参考或 Torch 对齐，固定案例必须全部通过 |
| `rmsnorm_reference_test` | 手算案例与一至三维、单特征输入的中心差分，验证独立解析梯度 |
| `linear_cpu_test` / `linear_cuda_test` | 二维 FP32 输入，有无 bias 的前向与反向；逐元素检查原始参数梯度及 bias 的 batch 求和；检查构造时拒绝错误的权重或偏置形状 |
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

## 数值精度检查

`precision_cpu_test` / `precision_cuda_test` 使用双参考逐元素验收：与数学参考误差达标，或者与 PyTorch 对齐，任一满足即通过。shape、dtype、device 必须先正确；原有布局、广播、整数语义、版本检查和自动微分结构测试继续保留。

```text
precision_test.cpp：生成样本 → 运行 mini-infer 前向、反向
                      ↓ 原始 dtype 编码、shape、stride、偏移、上游梯度
compare_torch.py：重建相同输入 → 计算 FP64 数学参考 + 实际运行 PyTorch
                      ↓
               逐元素验收 → 数学通过 / 仅 Torch 通过 / 两者均失败
```

### 安装一次，之后直接运行 CTest

PyTorch 只用于测试，不链接进 C++ 库。只要求环境能够导入 torch，不限制版本；已有 torch 无需重新安装。`tests/requirements-torch.txt` 仅声明 torch 依赖，实际版本记录在测试报告中。本地使用 Python 3.13.15 和 PyTorch 2.11.0+cpu，环境位于被 Git 忽略的 `build/torch-env`。

Windows，在已配置 C++ 编译器和 CMake 的终端运行：

```powershell
python -m venv build/torch-env
build/torch-env/Scripts/python.exe -m pip install -r tests/requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
cmake -S . -B build/tests -DBUILD_TESTING=ON
cmake --build build/tests --config Release --parallel 2
ctest --test-dir build/tests -C Release --output-on-failure
```

已有环境无需重复创建或安装。本机通过 uv 安装，也可以使用 `uv pip install --python build/torch-env/Scripts/python.exe -r tests/requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu`。

Linux / Colab 可直接使用已有 torch 的 Python 环境；需要新建时使用 `python3 -m venv build/torch-env`，环境内的 Python 路径为 `build/torch-env/bin/python`。GPU 对照需要支持 CUDA 的 torch；已有可用环境无需重装，否则选择与驱动兼容的 CUDA wheel，然后重新配置、构建 CUDA 测试。

CMake 优先查找上述项目内环境；也可传入 `-DPython3_EXECUTABLE=<python路径>`。找不到 Python 时，配置会警告，精度测试显示 `Skipped`；找到了 Python 但无法导入 torch，测试报错。没有可用 GPU 时，CUDA 精度测试显示 `Skipped`，不换成 CPU 参考。

当前本机只跑对照：

```powershell
ctest --test-dir build/source-layout-cpu -R '^precision_cpu_test$' --output-on-failure
```

CUDA 构建下使用 `-R '^precision_cuda_test$'`。CTest 自动导出样本并执行 PyTorch 对照，无需手动运行 C++ 导出程序。

### 样本与判定规则

- 覆盖 FP32、FP16、BF16 的 exp、sigmoid、SiLU、sqrt、rsqrt、neg、add、sub、mul、div，检查前向和所有输入梯度。
- 保留全部原有样本：每组最初的 513 个回归样本、确定性边界组合、每组 4,097 个固定种子随机样本。随机种子分别为 20261120（一元）和 20261121（二元）。共 102 组、884,818 次逐元素比较。
- 极端样本覆盖次正规数、正规数边界、最大有限值、正负号组合、相消、半 ULP 舍入边界、exp 溢出/下溢边界、SiLU 导数过零点，以及独立变化的极大、极小和零上游梯度。保留 `(a,b,g)=(60000,2,3)` 和 FP32/BF16 的 `(1e30,1e-10,1e-30)` 等反例。
- 输入先转换到被测 dtype，再导出原始位编码。PyTorch 恢复同样的 dtype、shape、stride、存储偏移及上游梯度，运行完整输入组，不把失败元素抽成小 Tensor 重算。CPU 对 CPU，CUDA 对 CUDA；不启用 autocast 或 compile。
- 一元、二元输入及上游梯度包含非连续视图；二元两侧使用不同 stride 和非零偏移。NaN、±Inf、±0 另测前向；二元特殊值共有 49 种输入组合。

每个元素的验收流程：

```text
先检查输出 shape、dtype、device，错误直接失败
    ↓
与舍入到目标 dtype 的数学参考比较
    ├─ 达标 → math_pass
    └─ 未达标 → 与同 dtype、同设备的 PyTorch 比较
                    ├─ 对齐 → torch_only_pass
                    └─ 不对齐 → failed
```

数学参考在 CPU 用 FP64 独立解析公式计算，不使用 autograd，也不复用 mini-infer 的算子。sigmoid 使用 `t=exp(-abs(x))`、导数 `t/(1+t)^2`，避免饱和输出的 `1-y` 丢失梯度；SiLU 等反向同样直接计算解析导数。中间结果保持 FP64，最终只舍入一次到目标 dtype。当前有限样本范围内，这些公式的中间量不会超出 FP64 表示范围；以后扩展输入范围或算子时需一并检查参考公式。

两种参考都使用相同的 dtype 容差。有限值按 `abs(actual-reference) <= atol + rtol*abs(reference)` 判断；同位置 NaN、同号 Inf 视为一致。零/非零和零符号差异仅作诊断，不额外设置失败条件。两种参考均不通过的元素仍记为 `failed`；是否导致 CTest 失败，由下面的分组规则决定，不修改单元素容差。

每个算子、dtype、前向及各输入梯度分别验收，各组不能互相抵消：

| 样本组 | 数量 | 验收要求 |
| --- | ---: | --- |
| basic（基础） | 513 | 全部通过 |
| boundary（边界） | 按算子和 dtype 变化；SiLU 为 720 | 全部通过，不计入随机通过率 |
| random（随机） | 4,097 | 通过率至少 99.5%，最多失败 20 个 |
| special（NaN/Inf/零等特殊值） | 一元 5、二元 49 | 全部通过 |

C++ 导出每组的 `cohorts` 数量，Python 检查分组覆盖完整样本且前后向一致。输入仍作为完整 Tensor 运行参考计算，只按下标分组统计，不改变样本顺序或向量化路径。随机组允许失败数用整数 `4097 // 200 = 20` 计算：20 个失败通过，21 个失败不通过；基础或边界组失败 1 个就不通过。结构性错误（shape/dtype/device）仍直接失败。

| dtype | rtol | atol |
| --- | ---: | ---: |
| FP32 | 1.3e-6 | 1e-5 |
| FP16 | 1e-3 | 1e-5 |
| BF16 | 1.6e-2 | 1e-5 |

容差采用 [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close) 的 dtype 默认表，前向和反向一致。数学误差达标可以独立通过；数学误差不达标但与 Torch 对齐，包括两边一起溢出或提前归零，也按约定通过。仅仅比 Torch 更接近数学值、但两项比较仍均超容差的样本不会自动放行。

`torch_comparator_test` 的 15 项自测覆盖按元素合并两种判定、两边都错、匹配与不匹配的 NaN/Inf、元数据错误、全部算子的前向/解析梯度、极端输入的参考公式、目标 dtype 舍入、位编码/布局、真实上游梯度以及缺失/重复分组；另覆盖三种 dtype 的随机组 20/21 个失败边界、严格组不能借用随机额度、分组元数据缺失或数量错误。

### 本机实测结果

2026-11-20，Windows CPU、Python 3.13.15、PyTorch 2.11.0+cpu，双参考规则下：

| 分类 | 元素数 |
| --- | ---: |
| 数学参考通过 | 879,677 |
| 数学参考未通过，但 Torch 对齐通过 | 5,141 |
| 两种参考均未通过 | 0 |
| 合计 | 884,818 |

全部 333,318 次前向及特殊值比较通过，所有反向比较也通过，`precision_cpu_test` 返回成功：

| 反向算子 | FP32 失败数 | FP16 失败数 | BF16 失败数 |
| --- | ---: | ---: | ---: |
| exp | 0 | 0 | 0 |
| sigmoid | 0 | 0 | 0 |
| SiLU | 0 | 0 | 0 |
| sqrt | 0 | 0 | 0 |
| rsqrt | 0 | 0 | 0 |
| add/sub/mul/div/neg（全部输入梯度） | 0 | 0 | 0 |

SiLU 反向改为 Torch 的 FP32 逐元素运算顺序后，原先 519 个失败全部修复。FP32/FP16/BF16 各 5,330 个反向样本中，数学参考分别通过 5,234/5,330/5,306 个，其余分别有 96/0/24 个通过 Torch 对齐。其他算子的对照结果未变；数值样本和容差未变。此前 Python 公式模拟残留 2 个 FP32 失败，本次以实际 C++ 导出结果为准，剩余 0 个。

此次 activation/autograd CPU 回归通过；新增 SiLU 回归覆盖低精度失败样本、非连续输入与上游梯度、跨 block 长度和零维标量。此前 sigmoid/rsqrt 回归仍适用。此次分组验收改动后，`precision_cpu_test` 和包含 15 项自测的 `torch_comparator_test` 均通过，884,818 次比较仍为 0 个失败。CUDA 反向实现已补齐，本机无可用 CUDA 编译器，尚未编译运行验证。

报告保存在 `<构建目录>/tests/torch-cpu/`（CUDA 为 `torch-cuda/`）：

- `samples.jsonl`：完整输入、上游梯度和 mini-infer 结果的位编码，以及 `cohorts` 分组数量。
- `report.json`：验收规则、环境版本、三类计数、两种参考诊断、全部失败下标和每组前 6 个失败样本。每个 cohort 记录下标起点、数量、通过率、允许失败数及 `accepted`；失败样本标注所属 cohort。`failed` 始终保留实际未对齐元素数，即使在随机额度内也不清零；`failed_groups` 统计未通过验收的算子/dtype/阶段组数，顶层 `accepted` 决定 CTest 返回码。

当前报告为 `build/source-layout-cpu/tests/torch-cpu/report.json`。本机无可用 CUDA 编译器，CUDA 双参考测试尚未运行。

## RMSNorm 组合算子测试

```powershell
ctest --test-dir build/source-layout-cpu -R '^rmsnorm' --output-on-failure
```

结构与精度测试分开：`rmsnorm_cpu_test` 检查精确可手算的前向、梯度与接口行为；`rmsnorm_precision_cpu_test` 用同一个可执行程序导出位编码，再用 FP64 独立解析公式和 `torch.nn.functional.rms_norm` 的真实 autograd 对照。复用逐元素测试的 dtype 容差与双参考规则，不修改已有 884,818 次逐元素比较的样本和报告。

10 个固定案例涵盖一维、二维、三维、非连续输入/权重/上游梯度、单特征、零输入、不同 eps、小输入、大输入以及 257 长度特征维。每种 dtype 都检查输出、输入梯度、跨所有前导维度累加的权重梯度。这里没有随机组，全部元素必须通过。

2026-11-20，Windows CPU、PyTorch 2.11.0+cpu，结构测试和参考公式的两项自测通过。精度结果如下，共 5,166 次比较、84 个失败，CTest 正常返回失败：

| dtype | 前向失败 | 输入梯度失败 | 权重梯度失败 |
| --- | ---: | ---: | ---: |
| FP32 | 0 | 0 | 0 |
| FP16 | 11 | 36 | 30 |
| BF16 | 0 | 5 | 2 |

例如 FP16 `large_input` 的 `x=-450`，平方 202500 写回 FP16 后成为 Inf，使归一化输出为零，数学参考和 Torch 约为 1.76953。其余失败也包括低精度反向舍入和相消。整段 FP32 计算模拟通过当前样本，但算子实现未改。

完整位编码与失败报告位于 `<构建目录>/tests/rmsnorm-cpu/`，CUDA 对应 `rmsnorm-cuda/`。CPU/CUDA 测试已在 CMake 注册；当前无可用 CUDA 编译器，GPU 结果尚未验证。Python 环境与逐元素精度测试共用，不锁 Torch 版本。

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

`matmul_cpu_test` 和 `matmul_cuda_test` 共用二维 matmul 用例。前向覆盖 8 种 dtype、非连续输入、低精度累加和整数回绕；CUDA 用例还覆盖输出元素数超过一个 block 的线程数。此前的前向版本已在 Colab 通过 15/15 项 CTest。

反向用例对 FP32、FP16、BF16 分别运行 `Y = X @ W.T`：X 为 [2,3]，原始权重 W 为 [4,3]，传入非均匀的上游梯度 G [2,4]。测试逐元素核对叶子梯度 `dX = G @ W` 和 `dW = G.T @ X`，并检查 shape、dtype、device。

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
