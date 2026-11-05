#include <stdexcept>

#include "ops/elementwise.h"

namespace mtensor {

// TODO：在本文件内实现模板 kernel，以及 FP16/BF16 对应的计算。

void LaunchAddKernel(
    const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    // TODO：按 dtype 分发、配置线程与线程块、启动 kernel 并检查错误。
    throw std::logic_error("LaunchAddKernel: not implemented");
}

void LaunchSubKernel(
    const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    // TODO：按 dtype 分发并启动对应 kernel。
    throw std::logic_error("LaunchSubKernel: not implemented");
}

void LaunchMulKernel(
    const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    // TODO：按 dtype 分发并启动对应 kernel。
    throw std::logic_error("LaunchMulKernel: not implemented");
}

void LaunchDivKernel(
    const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    // TODO：按 dtype 分发并启动对应 kernel。
    throw std::logic_error("LaunchDivKernel: not implemented");
}

void LaunchNegKernel(const void* a, void* output, std::size_t count, Dtype dtype) {
    // TODO：按 dtype 分发并启动对应 kernel。
    throw std::logic_error("LaunchNegKernel: not implemented");
}

}  // namespace mtensor
