# 测试

正确性测试在 `tests/CMakeLists.txt` 注册到 CTest。与 ggml 对比的性能程序由 `tests/benchmarks/CMakeLists.txt` 按需构建。

| 目录 | 内容 |
| --- | --- |
| `tensor/` | Tensor、布局、标量和 dtype |
| `tokenizer/` | 字节 tokenizer 和 BPE |
| `ops/` | 逐元素、激活、归约、矩阵乘和 Transformer 算子 |
| `modules/` | Linear 和 RMSNorm |
| `init/` | 参数初始化 |
| `models/` | GPT checkpoint |
| `autograd/` | CPU/CUDA 自动微分 |
| `optim/` | SGD、Adam 和训练回归 |
| `utils/` | 日志 |
| `precision/` | 逐元素精度样本生成、PyTorch 对照脚本和测试依赖 |
| `benchmarks/` | 与 ggml 的性能对比 |

三个 benchmark 源文件共用 `tests/benchmarks/bench_common.h` 中的计时、均值、中位数、预热和交替采样函数，各程序通过自己的执行回调确定计时范围。

需要支持 C++17 的编译器和 CMake 3.18+。从仓库根目录运行：

```powershell
cmake -S . -B build/tests -DBUILD_TESTING=ON
cmake --build build/tests --config Release --parallel 2
ctest --test-dir build/tests -C Release --output-on-failure
```

## 测试覆盖

| 测试程序 | 覆盖内容 |
| --- | --- |
| `tokenizer_test` | 字节词表顺序、编解码往返和词表恢复 |
| `bpe_tokenizer_test` | 合并顺序、同频选对、片段边界和规则重建 |
| `transformer_cpu_test` / `transformer_cuda_test` | 重复 token 梯度、索引上下界、RoPE 正逆旋转、Decoder 维度与前后向（输入及 8 份参数梯度）、极端 logits 的交叉熵和非连续输入 |
| `attention_cpu_test` / `attention_cuda_test` | FP64 前向参考、Q/K/V 梯度的中心差分、非连续输入和上游梯度、因果遮罩及输入版本检查 |
| `gpt_checkpoint_test` | 最终文件覆盖、词表恢复、参数和 logits 一致性、截断文件校验 |
| `sqrt_cpu_test` / `sqrt_cuda_test` | sqrt、rsqrt 的 8 种输入 dtype、整数提升、跨步视图、特殊值、浮点梯度和原地修改检测；rsqrt 低精度反向失败样本、不同 stride 的输入与上游梯度、跨 block 长度和零维标量 |
| `exp_cpu_test` / `exp_cuda_test` | exp 的 8 种输入 dtype、转置视图、跨 block 长度，以及复用输出的反向和版本检查 |
| `log_cpu_test` / `log_cuda_test` | log 的 8 种输入 dtype、转置视图、负值和零的行为、三种浮点 dtype 的非连续输入反向与版本检查 |
| `activation_cpu_test` / `activation_cuda_test` | sigmoid 和 SiLU 的浮点前向、梯度和转置输入；sigmoid 极端梯度的 FP32 中间计算、不同 stride 的输入与上游梯度、零维标量；SiLU 负尾部、计算分支边界、特殊值、no-grad 和输入版本检查；5 种整数输入提升及 sigmoid 输出版本检查；Softmax 的 8 种 dtype、大 logits、转置输入、归约维度、梯度及 FP16 长行 |
| `rmsnorm_cpu_test` / `rmsnorm_cuda_test` | Module 前向、输入/权重梯度、梯度累加、no-grad、错误形状和原地修改检测 |
| `rmsnorm_precision_cpu_test` / `rmsnorm_precision_cuda_test` | 三种浮点 dtype、10 个固定案例的前向与两个梯度，数学参考或 Torch 对齐，固定案例必须全部通过 |
| `rmsnorm_reference_test` | 手算案例与一至三维、单特征输入的中心差分，验证独立解析梯度 |
| `linear_cpu_test` / `linear_cuda_test` | 二维 FP32 输入，有无 bias 的前向与反向；逐元素检查原始参数梯度及 bias 的 batch 求和；检查构造时拒绝错误的权重或偏置形状 |
| `weight_init_test` | 固定 seed 下 Kaiming 均匀的 fan_in 边界及 Kaiming 正态的 fan_out 标准差 |
| `reduction_cpu_test` | sum 的 8 种 dtype、整数提升与回绕、非连续视图和端到端反向；max/min 的 8 种 dtype、维度、非连续视图、NaN 和并列极值反向 |
| `reduction_cuda_test` | sum 的 8 种 dtype、线程数边界、跨步循环、非连续视图和端到端反向；与 CPU 共用 max/min 用例 |
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

`tensor_cpu_test` 和 `tensor_cuda_test` 还覆盖 `to(Dtype)` 的非连续视图、浮点双向转换、整数转 FP32 和零维标量。CPU 用例另检查半 ULP 舍入、反向、同 dtype 共享存储和不支持的转换。

## 优化器测试

`optimizer_cpu_test` / `optimizer_cuda_test` 共用 `optim/optimizer_test.cpp`，覆盖 SGD、momentum、Adam 偏差修正、每个参数独立计步、梯度缺失与清零、共享参数原地更新、版本检测、错误参数和梯度校验。两个优化器还分别训练 `y = 2*x1 - 3*x2 + 0.5` 的线性回归模型，检查最终参数和 loss。

`optimizer_precision_cpu_test` / `optimizer_precision_cuda_test` 导出五组各 10 步的 FP32 参数更新轨迹，由 `precision/compare_optimizer.py` 同时对照同设备 PyTorch 和 CPU FP64 独立公式。每个设备有 250 次逐元素比较，要求两个参考都满足 `rtol=1e-5`、`atol=1e-6`。用例包含非连续参数、零维标量、延迟出现的梯度、零梯度、默认和自定义 beta，以及原始存储边界检查。

```powershell
ctest --test-dir build/tests -C Release -R '^optimizer' --output-on-failure
```

报告保存到 `<构建目录>/tests/optimizer-cpu/` 和 `optimizer-cuda/`，包含 `traces.jsonl` 与 `report.json`。PyTorch 依赖沿用下面的精度测试环境。

模型和优化器都保存参数的共享 Tensor 句柄。

```cpp
#include "mini_infer/modules/linear.h"
#include "mini_infer/optim/adam.h"

// x、target、weight、bias 已在选定设备上创建为 FP32 Tensor。
weight.requires_grad_(true);
bias.requires_grad_(true);
mtensor::Linear model(weight, bias);
mtensor::Adam optimizer({weight, bias}, 0.001f);

for (int step = 0; step < 500; ++step) {
    optimizer.zero_grad();
    auto error = model(x) - target;
    auto loss = (error * error).mean();
    loss.backward();
    optimizer.step();
}
```

SGD 的构造形式为 `SGD(parameters, learning_rate, momentum=0)`，Adam 为 `Adam(parameters, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)`。参数列表在构造时确定，参数须为非空、可写的 FP32 Tensor，相同设备上的重复起始地址会报错。非连续参数由现有 stride 算子和 `copy_()` 支持。优化器对象支持移动。

`step()` 在关闭建图的作用域内运行，先检查所有已存在梯度的 shape、dtype 和 device，再逐个更新参数及其状态。没有梯度的参数会跳过；已有零梯度的参数仍参与状态更新。`zero_grad()` 沿用 Tensor 的清零行为，保留已有梯度存储，动量和矩估计继续保存。

## 数值精度检查

`precision_cpu_test` / `precision_cuda_test` 先核对 shape、dtype、device，再逐元素比较数学参考和 PyTorch，满足任一容差即可通过。布局、广播、整数语义、版本检查和自动微分由对应的结构测试覆盖。

```text
precision/precision_test.cpp：生成样本 → 运行 mini-infer 前向、反向
                      ↓ 原始 dtype 编码、shape、stride、偏移、上游梯度
precision/compare_torch.py：重建相同输入 → 计算 FP64 数学参考 + 实际运行 PyTorch
                      ↓
               逐元素验收 → 数学通过 / 仅 Torch 通过 / 两者均失败
```

### 安装测试依赖

PyTorch 是测试依赖。`tests/precision/requirements-torch.txt` 声明 torch 依赖，实际版本由测试报告记录。已有可导入 torch 的 Python 环境可直接使用。以下示例使用 `build/torch-env`，Windows CPU 历史结果来自 Python 3.13.15 和 PyTorch 2.11.0+cpu。

Windows，在已配置 C++ 编译器和 CMake 的终端运行：

```powershell
python -m venv build/torch-env
build/torch-env/Scripts/python.exe -m pip install -r tests/precision/requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
cmake -S . -B build/tests -DBUILD_TESTING=ON
cmake --build build/tests --config Release --parallel 2
ctest --test-dir build/tests -C Release --output-on-failure
```

也可用 `uv pip install --python build/torch-env/Scripts/python.exe -r tests/precision/requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu` 安装依赖。

Linux / Colab 可用 `python3 -m venv build/torch-env` 创建环境，其中 Python 位于 `build/torch-env/bin/python`。GPU 对照需要支持 CUDA 且与驱动兼容的 torch。

CMake 优先使用 `build/torch-env`，也可通过 `-DPython3_EXECUTABLE=<python路径>` 指定。PyTorch 对照需要该环境能导入 `torch`。缺少 Python 或可用 GPU 时，受影响的对照测试会跳过。

单跑 CPU 精度对照：

```powershell
ctest --test-dir build/tests -C Release -R '^precision_cpu_test$' --output-on-failure
```

CUDA 构建下使用 `-R '^precision_cuda_test$'`。CTest 负责导出样本并执行 PyTorch 对照。

### 样本与判定规则

- 覆盖 FP32、FP16、BF16 的 exp、sigmoid、SiLU、sqrt、rsqrt、neg、add、sub、mul、div，检查前向和所有输入梯度。
- 样本包括每组最初的 513 个回归样本、确定性边界组合、每组 4,097 个固定种子随机样本。随机种子分别为 20261120（一元）和 20261121（二元）。共 102 组、884,818 次逐元素比较。
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

数学参考在 CPU 用 FP64 解析公式计算，最后舍入到目标 dtype。sigmoid 梯度用 `t/(1+t)^2`，其中 `t=exp(-abs(x))`，避免前向饱和后 `1-y` 丢失梯度。

两种参考使用相同的 dtype 容差。有限值按 `abs(actual-reference) <= atol + rtol*abs(reference)` 判断。同位置 NaN、同号 Inf 视为一致。零/非零和零符号差异用于诊断。两种参考均未通过的元素记为 `failed`，CTest 根据分组阈值决定结果。

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

前向和反向均采用 PyTorch 的 dtype 默认容差。[torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)

`torch_comparator_test` 的 15 项自测覆盖按元素合并两种判定、两边都错、匹配与不匹配的 NaN/Inf、元数据错误、全部算子的前向/解析梯度、极端输入的参考公式、目标 dtype 舍入、位编码/布局、真实上游梯度以及缺失/重复分组；另覆盖三种 dtype 的随机组 20/21 个失败边界、严格组不能借用随机额度、分组元数据缺失或数量错误。

### 本机实测结果

2026-11-20，Windows CPU、Python 3.13.15、PyTorch 2.11.0+cpu，双参考规则下：

| 分类 | 元素数 |
| --- | ---: |
| 数学参考通过 | 879,677 |
| 数学参考未通过，但 Torch 对齐通过 | 5,141 |
| 两种参考均未通过 | 0 |
| 合计 | 884,818 |

其中 333,318 次是前向和特殊值比较。SiLU 反向的 519 个失败由 FP32 运算顺序修复。改动和复现样本见[算子精度笔记](../note/code/算子精度问题.md)。

本机后续 CPU 和 CUDA 回归各完成 884,818 次比较，均为零失败。`torch_comparator_test` 的 15 项自测也通过。

报告保存在 `<构建目录>/tests/torch-cpu/`（CUDA 为 `torch-cuda/`）：

- `samples.jsonl`：完整输入、上游梯度和 mini-infer 结果的位编码，以及 `cohorts` 分组数量。
- `report.json`：验收规则、环境版本、三类计数、两种参考诊断、全部失败下标和每组前 6 个失败样本。每个 cohort 记录下标起点、数量、通过率、允许失败数及 `accepted`；失败样本标注所属 cohort。`failed` 始终保留实际未对齐元素数，即使在随机额度内也不清零；`failed_groups` 统计未通过验收的算子/dtype/阶段组数，顶层 `accepted` 决定 CTest 返回码。

## RMSNorm 测试

```powershell
ctest --test-dir build/tests -C Release -R '^rmsnorm' --output-on-failure
```

| 测试 | 检查内容 |
| --- | --- |
| `rmsnorm_cpu_test`、`rmsnorm_cuda_test` | 手算的前向和梯度、层接口、累加、形状和原地修改 |
| `rmsnorm_precision_cpu_test`、`rmsnorm_precision_cuda_test` | 导出位编码，对照 FP64 解析公式和 `torch.nn.functional.rms_norm` 的实际梯度 |

10 个固定案例覆盖不同维数和 stride、单特征、零值、eps、大小输入及特征维长度 257。每种 dtype 都比较输出、输入梯度和权重梯度，所有元素必须通过。

FP16/BF16 改用 FP32 中间量后，CPU 和 CUDA 各 5,166 次比较均通过。报告位于 `<构建目录>/tests/rmsnorm-cpu/` 和 `rmsnorm-cuda/`。[失败样本与修复](../note/code/算子精度问题.md#8-补测-rmsnorm基础算子全过组合仍可能出错)

## 布局、迁移和梯度

| 测试 | 重点 |
| --- | --- |
| `tensor_cuda_test` | 八种 dtype 双向迁移、位编码保留、空对象和标量、共享存储生命周期 |
| `autograd_cpu_test`、`autograd_cuda_test` | 视图反向、梯度清零与累加、写入版本、更新参数后重新前向 |
| `reduction_cpu_test`、`reduction_cuda_test` | 全元素和按维度归约、广播前向和双边梯度、非连续输入 |
| `layout_cpu_test` | 形状、stride、视图共享和布局反向 |

只跑其中一组，例如 Tensor：

```powershell
ctest --test-dir build/tests -C Release -R "^tensor_cpu_test$" --output-on-failure
```

CMake 检测到 Toolkit >= 12.4 和可用 CUDA 编译器时设置 `ENABLE_CUDA`，否则构建 CPU 版。更换工具链时用新构建目录重新检测。GPU 或驱动不可用时跳过 CUDA 测试。

CTest 从构建目录下的 `tests/` 运行正确性测试。默认 Logger 在进程工作目录生成 `.log`。

在有 CUDA Toolkit 和 GPU 的 Colab 环境，从已同步最新源码的仓库根目录运行：

```bash
bash ./test.sh
```

Windows + RTX 4060 可在 VS Code 的 Git Bash 终端运行同一命令。脚本默认使用 CUDA 架构 `80` 并保留 PTX，适用于 Ampere 及更新 GPU。CMake 从环境选择编译器，生成器可通过 `CMAKE_GENERATOR` 指定。Windows CUDA 构建需要 Toolkit 和受支持的 MSVC C++ 工具链。本机使用被 Git 忽略的 `cmake/windows-cuda.cmake`，从 `CUDA_PATH` 读取 VS 集成文件，补齐未注册的 CUDA toolset。CMake 在该文件存在时自动加载。

其他 GPU 可在运行前指定架构，例如 `CUDA_ARCH=80 bash ./test.sh`。也可以手动执行下面的 CMake/CTest 命令：

```bash
cmake -S . -B build/tests -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/tests --config Release --parallel 2
ctest --test-dir build/tests -C Release --output-on-failure
```

上面的 `89` 对应 Colab L4 和 RTX 4060；其他 GPU 应改成其对应架构。

VS Code 安装 C/C++ 与 CMake Tools 后运行 `CMake: Configure`，由 CMake Tools 提供实际编译器、include 和宏定义；`.vscode/` 不保存机器绝对路径。未安装完整 Toolkit 时，可选的 `build/cuda-headers-12.4/include` 仅供本地补全，不参与 CUDA 构建。首次配置如需选择 Kit，Windows CUDA 开发选择 Visual Studio x64。

## Matmul 正确性与 CPU 性能对比

`matmul_cpu_test` 和 `matmul_cuda_test` 共用二维 matmul 用例。前向覆盖 8 种 dtype、非连续输入、低精度累加和整数回绕；CUDA 用例还覆盖输出元素数超过一个 block 的线程数。

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

性能报告记录两侧版本、硬件、编译选项、线程数、dtype、M/K/N、布局及计时范围。

## CUDA matmul 性能对比

在 Colab GPU runtime 按顺序运行 `setup.ipynb`。CMake 单元格根据 GPU 计算能力选择 CUDA 架构，在 `build/cuda-colab` 一次构建 Release 版测试和 benchmark。接下来的 CTest 和 CUDA perf 单元格分别运行已构建的程序。

默认测 `M=1/32/128, K=N=1024`。在最后一个单元格的命令末尾加 `--model`，可运行表中的 13 个模型层尺寸。当前单 block 路径跑大尺寸耗时较长。

两侧在同一张 GPU 上使用相同的 FP32 输入和 `weight.T` 布局。ggml 运行单个 `ggml_mul_mat` 节点，采用 llama.cpp 的默认 CUDA Graph 配置和 kernel 分发。预分配、上传和输出核对在计时外。预热 3 次后，两侧交替测量各 10 次，计时包含提交和等待 GPU 完成，输出均值、中位数、GFLOP/s 和耗时比。

## FP32 前向完整调用对比

`ops_cpu_bench` 和 `ops_cuda_bench` 共用 `ops_bench.cpp`，覆盖 FP32 的 17 组前向接口。默认输入形状为 [1,1024]、[32,1024] 和 [128,1024]，每个设备共 51 组结果。matmul 和 Linear 的权重为 [1024,1024]，Linear 带偏置，RMSNorm 带权重。

| mini-infer 接口 | ggml 对应计算 |
| --- | --- |
| add、sub、mul、div、neg | 同名算子 |
| sqrt、exp、sigmoid、silu | 同名算子 |
| rsqrt | 全 1 Tensor 除以 sqrt(x) |
| sum_all、sum_last | sum、sum_rows |
| mean_all、mean_last | 展平后 mean、按行 mean |
| matmul | mul_mat，权重 K 维连续 |
| Linear | mul_mat 后广播加 bias |
| RMSNorm | rms_norm 后广播乘 weight |

计时覆盖两端的一次完整前向调用，包括输出和临时存储分配、ggml 图构建、计算、等待完成及临时对象释放。mini-infer 在 `NogradNode` 下运行。输入、权重、上传、后端初始化和 Linear 对象创建均在计时外。ggml 每次调用重新创建输出 context、图和输出 buffer，后端在整个程序中复用，并保留上游默认的优化和缓存行为。

CPU 的 mini-infer 使用当前单线程实现，ggml 默认线程配置为 4，可用 `--ggml-threads` 指定。两边先用同一输入校验输出，再预热并交替计时，最后重新校验输出。校验同时检查形状、dtype、元素数和有限值，误差要求为 `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`。校验失败时程序返回失败。ggml 后端不支持的计算会输出 `unsupported_ggml`，耗时字段留空。

```powershell
cmake -S . -B build/ops-bench -DBUILD_BENCHMARKS=ON -DBUILD_CUDA_BENCHMARKS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/ops-bench --config Release --target ops_cpu_bench ops_cuda_bench --parallel 4
build/ops-bench/tests/benchmarks/Release/ops_cpu_bench.exe --samples 20 --warmups 5 | Set-Content -Encoding UTF8 build/ops-bench/ops-cpu.csv
build/ops-bench/tests/benchmarks/Release/ops_cuda_bench.exe --samples 20 --warmups 5 | Set-Content -Encoding UTF8 build/ops-bench/ops-cuda.csv
```

CPU 单独构建时使用 `-DBUILD_CUDA_BENCHMARKS=OFF` 并仅指定 `ops_cpu_bench` 目标。Linux 单配置构建使用 `-DCMAKE_BUILD_TYPE=Release`，程序位于构建目录的 `tests/benchmarks/`。

可用 `--op rmsnorm --rows 32 --cols 4096` 单独运行指定算子和形状。其余算子名称见 CSV 的 `op` 字段，默认 `--op all`、3 次预热、10 次采样。`--cols` 同时指定矩阵乘的 K 和 N。

CSV 开头以 `#` 记录计时范围、版本、构建配置、编译器、线程或 GPU 信息。数据行包含均值、中位数、最大绝对误差和 `mini_over_ggml`。这个比值是 mini-infer 耗时除以 ggml 耗时，大于 1 表示 mini-infer 用时更多。

## 待办

- [ ] 增加内存分配、设备复制的故障注入和泄漏检查。
