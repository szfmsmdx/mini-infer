# 网络模块

`Module::operator()` 调用虚函数 `forward`，所以可以把层当函数用：

```cpp
#include "mini_infer/modules/linear.h"

// x: [batch, in]，weight: [out, in]，bias: [out]
mtensor::Linear projection(weight, bias);  // 无偏置时写 projection(weight)
auto y = projection(x);                    // [batch, out]
```

`Linear` 保存传入的 Tensor 句柄，不复制权重数据，也不改变参数的 `requires_grad`。需要保留梯度时，在权重和偏置上设置 `requires_grad_(true)`；反向结束后可从原始 Tensor 读取梯度。

[`Linear::forward`](linear.cpp) 用现有算子计算 `x @ weight.T`，有偏置时再沿 batch 广播相加。CPU 和 CUDA 共用[同一组测试](../../tests/linear_test.cpp)，包括对原始权重和偏置梯度的检查。

## RMSNorm

`RMSNorm(weight, eps)` 保存一维权重，`forward(x)` 调用 `rms_norm(x, weight, eps)`，沿最后一维归一化。权重长度在算子入口与输入最后一维检查；反向复用平方、mean、rsqrt、乘法和广播的计算图。

CPU/CUDA 共用 [`rmsnorm_test.cpp`](../../tests/rmsnorm_test.cpp)。结构测试检查 Module/算子入口、参数梯度累加、no-grad、形状检查和原地修改检测；精度测试由 [`compare_rmsnorm.py`](../../tests/compare_rmsnorm.py) 对照 FP64 解析公式及同设备 Torch，检查输出、输入梯度和权重梯度。

2026-11-20 本机 CPU：结构测试通过；5,166 次精度比较中 FP32 全部通过，FP16 有 77 个失败，BF16 有 7 个失败。FP16 大输入平方写回时溢出，低精度中间结果也影响反向；当前组合实现尚未修复。CUDA 测试已注册，尚未编译运行验证。
