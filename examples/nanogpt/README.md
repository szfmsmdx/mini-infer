# 训练和加载字符级 GPT

`train_gpt` 用 Tiny Shakespeare 从零训练两层 GPT。模型用 RoPE、因果注意力、RMSNorm 和 SiLU，词嵌入与输出投影共享权重。[GPT](../../src/models/gpt.cpp) · [SimpleDecoderLayer](../../src/layers/SimpleDecoderLayer.cpp) · [RoPE](../../src/ops/position_embedding/rope.cpp)

## 构建与运行

从仓库根目录运行以下 PowerShell 命令。CMake 检测到 CUDA Toolkit 后，两个目标均可使用 GPU。

```powershell
python examples/nanogpt/prepare_shakespeare.py
cmake -S . -B build/gpt -DBUILD_TESTING=OFF
cmake --build build/gpt --config Release --target train_gpt chat_gpt
New-Item -ItemType Directory -Force build/gpt-run | Out-Null
Set-Location build/gpt-run
..\gpt\Release\train_gpt.exe ..\gpt-data\input.txt 1600 cuda 4
..\gpt\Release\chat_gpt.exe .\gpt_final.bin cuda
```

## 训练输出

| 文件 | 内容 |
| --- | --- |
| `train_gpt.log` | 每步 loss 和耗时，定期输出固定窗口上的训练和验证 loss |
| `gpt_final.bin` | 模型配置、字节词表和 FP32 权重。再次成功保存时覆盖此文件 |

语料按顺序取前 90% 训练、后 10% 验证。当前模型的上下文窗口是 32 个 token。

## 加载与续写

在提示符输入英文，模型会按训练语料续写。输入 `/quit` 退出。也可以传入提示词，生成一次后退出。

```powershell
..\gpt\Release\chat_gpt.exe .\gpt_final.bin cuda "ROMEO:"
```

两个程序都接受 `cpu`。Linux 的单配置构建使用 `-DCMAKE_BUILD_TYPE=Release`，可执行文件位于 `build/gpt/`。
