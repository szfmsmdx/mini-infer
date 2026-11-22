# 网络模块

`Module::operator()` 调用虚函数 `forward`，所以可以把层当函数用：

```cpp
#include "mini_infer/modules/linear.h"

// x: [batch, in]，weight: [out, in]，bias: [out]
mtensor::Linear projection(weight, bias);  // 无偏置时写 projection(weight)
auto y = projection(x);                    // [batch, out]
```

`Linear` 保存传入的 Tensor 句柄，不复制权重数据，也不改变参数的 `requires_grad`。需要保留梯度时，在权重和偏置上设置 `requires_grad_(true)`；反向结束后可从原始 Tensor 读取梯度。

[`Linear::forward`](linear.cpp) 用现有算子计算 `x @ weight.T`，有偏置时再沿 batch 广播相加。CPU 和 CUDA 共用[同一组测试](../../tests/modules/linear_test.cpp)，包括对原始权重和偏置梯度的检查。

## RMSNorm

`RMSNorm(weight, eps)` 保存一维权重，`forward(x)` 调用 `rms_norm(x, weight, eps)`，沿最后一维归一化。算子入口检查权重长度与输入最后一维是否一致。FP32 使用组合运算，FP16/BF16 使用专用前向和反向，并保留 FP32 中间量。

CPU/CUDA 共用 [`rmsnorm_test.cpp`](../../tests/modules/rmsnorm_test.cpp)。结构测试检查 Module/算子入口、参数梯度累加、no-grad、形状检查和原地修改检测；精度测试由 [`compare_rmsnorm.py`](../../tests/precision/compare_rmsnorm.py) 对照 FP64 解析公式及同设备 Torch，检查输出、输入梯度和权重梯度。

本机 CPU 和 CUDA 的结构测试均通过，每个设备的 5,166 次精度比较也全部通过。
