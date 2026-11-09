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
    Tensor out(a.shape(), a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        AddCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
               a.shape(), a.stride(), b.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        AddCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(), a.dtype(),
                a.shape_cuda(), a.stride_cuda(), b.stride_cuda(), a.dim());
#else
        throw std::logic_error("add: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("add: unsupported device");
    }
    return out;
}
Tensor sub(const Tensor& a, const Tensor& b){
    if(a.empty() || b.empty()){
        throw std::invalid_argument("Get Empty Tensor");
    }
    CheckElementwiseInputs(a, b);
    Tensor out(a.shape(), a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        SubCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
               a.shape(), a.stride(), b.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        SubCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(), a.dtype(),
                a.shape_cuda(), a.stride_cuda(), b.stride_cuda(), a.dim());
#else
        throw std::logic_error("sub: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("sub: unsupported device");
    }
    return out;
}
Tensor mul(const Tensor& a, const Tensor& b){
    if(a.empty() || b.empty()){
        throw std::invalid_argument("Get Empty Tensor");
    }
    CheckElementwiseInputs(a, b);
    Tensor out(a.shape(), a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        MulCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
               a.shape(), a.stride(), b.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        MulCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(), a.dtype(),
                a.shape_cuda(), a.stride_cuda(), b.stride_cuda(), a.dim());
#else
        throw std::logic_error("mul: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("mul: unsupported device");
    }
    return out;
}
Tensor div(const Tensor& a, const Tensor& b){
    if(a.empty() || b.empty()){
        throw std::invalid_argument("Get Empty Tensor");
    }
    CheckElementwiseInputs(a, b);
    Tensor out(a.shape(), a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        DivCPU(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.dtype(),
               a.shape(), a.stride(), b.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        DivCUDA(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(), a.dtype(),
                a.shape_cuda(), a.stride_cuda(), b.stride_cuda(), a.dim());
#else
        throw std::logic_error("div: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("div: unsupported device");
    }
    return out;
}
Tensor neg(const Tensor& a){
    if(a.empty()){
        throw std::invalid_argument("Get Empty Tensor");
    }
    Tensor out(a.shape(), a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        NegCPU(a.data_ptr(), out.data_ptr(), a.dtype(), a.shape(), a.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        NegCUDA(a.data_ptr(), out.data_ptr(), a.numel(), a.dtype(),
                a.shape_cuda(), a.stride_cuda(), a.dim());
#else
        throw std::logic_error("neg: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("neg: unsupported device");
    }
    return out;
}

}  // namespace mtensor
