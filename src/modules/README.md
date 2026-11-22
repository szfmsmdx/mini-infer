# 网络模块

给出输入和输出维度，`Linear` 就会创建自己的参数：

```cpp
#include "mini_infer/modules/linear.h"

mtensor::Linear projection(128, 256, /*bias=*/false);
auto y = projection(x);                  // [batch, 128] -> [batch, 256]
auto weights = projection.parameters();  // 一份 [256, 128] 的权重
```

按尺寸构造时，构造函数调用 `Linear::init()` 一次，创建并登记 weight 和可选的 bias。参数默认在 CPU 上，使用 FP32 并保存梯度。默认权重使用 `kaiming_uniform_`，取 `a = sqrt(5)` 和 `fan_in = in_features`，采样区间仍是 `±1 / sqrt(in_features)`；偏置默认为开启，初值为 0。可以用第四个参数指定设备。

[`weight_init.h`](../init/weight_init.h) 的 `normal_` 和 `uniform_` 接收分布参数。`kaiming_normal_` 和 `kaiming_uniform_` 则根据 `fan_in` 或 `fan_out` 及 Leaky ReLU 斜率计算采样尺度；斜率为 0 时对应 ReLU。它们原地填充主机 FP32 缓冲区，由层负责创建 Tensor。

`Module` 声明 `init()`，由各层的构造函数决定何时调用。`Linear(weight, bias)` 和 `Linear(weight)` 只登记传入的 Tensor，不重跑初始化，也不改变 `requires_grad`。

`Module::parameters()` 先收集当前层直接登记的参数，再按组合层指定的顺序遍历子层。`Linear` 按 weight、bias 的顺序登记。返回的 Tensor 句柄共享原参数的数据和梯度，原地修改会影响下一次前向。

`Module::operator()` 调用虚函数 `forward`。[`Linear::forward`](linear.cpp) 计算 `x @ weight.T`，有偏置时再沿 batch 广播相加。CPU 和 CUDA 共用[同一组测试](../../tests/modules/linear_test.cpp)。

## RMSNorm

`RMSNorm(channels, eps, device)` 创建一份初值为 1 的 FP32 权重。`RMSNorm(weight, eps)` 仍可传入已有的一维权重，长度须等于输入最后一维。`forward(x)` 沿最后一维归一化。FP32 复用基础算子，FP16/BF16 的前向和反向使用专用实现与 FP32 中间量。[RMSNorm 实现](rmsnorm.cpp) · [测试说明](../../tests/README.md#rmsnorm-测试)

## 组合层

`Embedding(vocab_size, channels, device)` 持有 `[vocab_size, channels]` 权重，前向复用 `embedding()` 查表。RoPE 没有可训练参数，`SimpleDecoderLayer` 直接调用 [`rope()`](../ops/position_embedding/rope.cpp) 旋转 Q 和 K。

`SimpleDecoderLayer` 按通道数和头数构造子层，MLP 在构造列表里声明 `C → 4C → C`。`GPT` 组合 Embedding、多层 decoder 和最终 RMSNorm。GPT 只指定标准差：Embedding、Q/K/V 和 MLP expand 用 `0.02`，两个残差输出投影用 `0.02 / sqrt(2 × 层数)`。各层使用 [`weight_init.cpp`](../init/weight_init.cpp) 中的采样实现，按构造顺序消费同一个由 GPT 的 seed 初始化的生成器。

`ModelConfig` 保存公共模型维度。创建 GPT 时，`GPTConfig` 在分配权重前检查词表大小，以及 RoPE 和 MLP 对维度的要求。直接创建 `SimpleDecoderLayer` 时也会检查后两项。

checkpoint 按顺序存储参数：Embedding → 每个 decoder 层的 attention norm、Q、K、V、attention projection、MLP norm、expand、contract → 最终 RMSNorm。输出投影复用 Embedding 权重，只登记一次，所以 `L` 层共有 `8L + 2` 份参数。

[SimpleDecoderLayer](../layers/SimpleDecoderLayer.cpp) · [Embedding 层](../layers/embedding.cpp) · [词嵌入算子](../ops/embedding/embedding.cpp) · [GPT](../models/gpt.cpp)
