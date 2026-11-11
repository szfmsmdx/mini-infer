#include <memory>
#include <stdexcept>
#include <utility>

#include "ops/elementwise.h"
#include "autograd/autograd.h"

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

bool IsGradDtype(Dtype dtype){
    return dtype == Dtype::kBFloat16 || 
            dtype == Dtype::kFloat16 || 
            dtype == Dtype::kFloat32;
}

// -------------------- backward node --------------------
class AddNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& g) override {
        return { g, g };
    }
};

class SubNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& g) override {
        return { g, neg(g) };
    }
};

class NegNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& g) override {
        return { neg(g) };
    }
};

class MulNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& g) override {
        return { mul(g, inputs[1]), mul(g, inputs[0]) };
    }
};

class DivNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& g) override {
        return { div(g, inputs[1]),
                 neg(div(mul(g, inputs[0]), mul(inputs[1], inputs[1]))) };
    }
};

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

    // 反向传播建图
    if(EnableGrad() && IsGradDtype(a.dtype())) {
        auto node = std::make_shared<AddNode>();
        node->inputs = { a, b };
        out.set_grad_node(std::move(node));
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

    if (EnableGrad() && IsGradDtype(a.dtype())) {
        auto node = std::make_shared<SubNode>();
        node->inputs = { a, b };
        out.set_grad_node(std::move(node));
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

    if (EnableGrad() && IsGradDtype(a.dtype())) {
        auto node = std::make_shared<MulNode>();
        node->inputs = { a, b };
        out.set_grad_node(std::move(node));
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

    if (EnableGrad() && IsGradDtype(a.dtype())) {
        auto node = std::make_shared<DivNode>();
        node->inputs = { a, b };
        out.set_grad_node(std::move(node));
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

    if (EnableGrad() && IsGradDtype(a.dtype())) {
        auto node = std::make_shared<NegNode>();
        node->inputs = { a };
        out.set_grad_node(std::move(node));
    }

    return out;
}

}  // namespace mtensor
