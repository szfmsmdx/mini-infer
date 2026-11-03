#include <stdexcept>

#include "ops/elementwise.h"

namespace mtensor {
namespace {

void CheckElementwiseInputs(const Tensor& a, const Tensor& b) {
    if(a.device() != b.device()){
        throw std::invalid_argument("Tensor must be same device");
    }else if(a.dtype() != b.dtype()){
        throw std::invalid_argument("Tensor must be same dtype");
    }else if(a.shape() != b.shape()){
        throw std::invalid_argument("Tensor must be same size");
    }
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b){
    if(a.empty() || b.empty()){
        throw std::invalid_argument("Get Empty Tensor");
    }
    CheckElementwiseInputs(a, b);
    Tensor out = a.clone();
    // TODO：分配构造函数实现后，直接分配输出，替换 clone()。
    if (a.device() == Device::kCPU) {
        AddCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(), a.dtype());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        LaunchAddKernel(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(), a.dtype());
#else
        throw std::logic_error("add: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("add: unsupported device");
    }
    return out;
}
Tensor sub(const Tensor& a, const Tensor& b){
    // TODO：检查输入、分配输出并按 device 调用后端入口。
    throw std::logic_error("sub: not implemented");
}
Tensor mul(const Tensor& a, const Tensor& b){
    // TODO：检查输入、分配输出并按 device 调用后端入口。
    throw std::logic_error("mul: not implemented");
}
Tensor div(const Tensor& a, const Tensor& b){
    // TODO：检查输入、分配输出并按 device 调用后端入口。
    throw std::logic_error("div: not implemented");
}

}  // namespace mtensor
