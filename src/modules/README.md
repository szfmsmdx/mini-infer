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
