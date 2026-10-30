#pragma once

#include <cstddef>

namespace mtensor {

void* AllocateCPU(std::size_t bytes);

// TODO: 补充 AllocateCUDA / FreeCUDA，供 Tensor 构造、clone 和 storage_ 删除器使用。

}  // namespace mtensor
