#include <cstddef>
#include <stdexcept>

#include "ops/elementwise.h"

namespace mtensor {
namespace {

// 从原 add() 移来的循环；由本文件的 CPU 入口选择 T 后调用。
template <typename T>
void AddTypedCPU(const T* a, const T* b, T* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = a[i] + b[i];
    }
}

// TODO：在本文件内实现 SubTypedCPU / MulTypedCPU / DivTypedCPU。

}  // namespace

void AddCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    // TODO：单独处理 FP16/BF16；其余类型用 DISPATCH_DTYPE 调用 AddTypedCPU<T>()。
    throw std::logic_error("AddCPU: dtype dispatch not implemented");
}

void SubCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    // TODO：按 dtype 调用对应的计算函数。
    throw std::logic_error("SubCPU: not implemented");
}

void MulCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    // TODO：按 dtype 调用对应的计算函数。
    throw std::logic_error("MulCPU: not implemented");
}

void DivCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    // TODO：按 dtype 调用对应的计算函数。
    throw std::logic_error("DivCPU: not implemented");
}

}  // namespace mtensor
