#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ops/elementwise.h"
#include "ops/backend.h"
#include "tensor/check.h"
#include "autograd/autograd.h"
#include "autograd/backward_nodes/elementwise.h"

namespace mtensor {
namespace {

std::vector<int> BroadcastShape(const std::vector<int>& a, const std::vector<int>& b) {
    const std::size_t ndim = a.size() > b.size() ? a.size() : b.size();
    std::vector<int> shape(ndim, 1);
    for (std::size_t i = 0; i < ndim; ++i) {
        const int a_size = i < a.size() ? a[a.size() - 1 - i] : 1;
        const int b_size = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (a_size != b_size && a_size != 1 && b_size != 1) {
            throw std::invalid_argument("BroadcastShape: incompatible shapes");
        }
        shape[ndim - 1 - i] = a_size == 1 ? b_size : a_size;
    }
    return shape;
}

}  // namespace


Tensor add(const Tensor& a, const Tensor& b){
    CheckNotEmpty(a, "Get Empty Tensor");
    CheckNotEmpty(b, "Get Empty Tensor");
    const auto shape = BroadcastShape(a.shape(), b.shape());
    Tensor lhs = a.shape() == shape ? a : a.broadcast(shape);
    Tensor rhs = b.shape() == shape ? b : b.broadcast(shape);
    CheckSameDevice(lhs, rhs, "Tensor must be same device");
    CheckSameDtype(lhs, rhs, "Tensor must be same dtype");
    CheckSameShape(lhs, rhs, "Tensor must be same size");
    Tensor out(shape, a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        AddCPU(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), a.dtype(),
               shape, lhs.stride(), rhs.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        AddCUDA(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel(), a.dtype(),
                out.shape_cuda(), lhs.stride_cuda(), rhs.stride_cuda(), out.dim());
#else
        throw std::logic_error("add: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("add: unsupported device");
    }

    // 反向传播建图
    if(EnableGrad() && CheckGradDtype(a.dtype())) {
        auto node = std::make_shared<AddNode>();
        node->inputs = { lhs, rhs };
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor sub(const Tensor& a, const Tensor& b){
    CheckNotEmpty(a, "Get Empty Tensor");
    CheckNotEmpty(b, "Get Empty Tensor");
    const auto shape = BroadcastShape(a.shape(), b.shape());
    Tensor lhs = a.shape() == shape ? a : a.broadcast(shape);
    Tensor rhs = b.shape() == shape ? b : b.broadcast(shape);
    CheckSameDevice(lhs, rhs, "Tensor must be same device");
    CheckSameDtype(lhs, rhs, "Tensor must be same dtype");
    CheckSameShape(lhs, rhs, "Tensor must be same size");
    Tensor out(shape, a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        SubCPU(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), a.dtype(),
               shape, lhs.stride(), rhs.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        SubCUDA(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel(), a.dtype(),
                out.shape_cuda(), lhs.stride_cuda(), rhs.stride_cuda(), out.dim());
#else
        throw std::logic_error("sub: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("sub: unsupported device");
    }

    if (EnableGrad() && CheckGradDtype(a.dtype())) {
        auto node = std::make_shared<SubNode>();
        node->inputs = { lhs, rhs };
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor mul(const Tensor& a, const Tensor& b){
    CheckNotEmpty(a, "Get Empty Tensor");
    CheckNotEmpty(b, "Get Empty Tensor");
    const auto shape = BroadcastShape(a.shape(), b.shape());
    Tensor lhs = a.shape() == shape ? a : a.broadcast(shape);
    Tensor rhs = b.shape() == shape ? b : b.broadcast(shape);
    CheckSameDevice(lhs, rhs, "Tensor must be same device");
    CheckSameDtype(lhs, rhs, "Tensor must be same dtype");
    CheckSameShape(lhs, rhs, "Tensor must be same size");
    Tensor out(shape, a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        MulCPU(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), a.dtype(),
               shape, lhs.stride(), rhs.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        MulCUDA(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel(), a.dtype(),
                out.shape_cuda(), lhs.stride_cuda(), rhs.stride_cuda(), out.dim());
#else
        throw std::logic_error("mul: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("mul: unsupported device");
    }

    if (EnableGrad() && CheckGradDtype(a.dtype())) {
        auto node = std::make_shared<MulNode>();
        node->inputs = { lhs, rhs };
        node->input_versions[0] = lhs.version();
        node->input_versions[1] = rhs.version();
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor div(const Tensor& a, const Tensor& b){
    CheckNotEmpty(a, "Get Empty Tensor");
    CheckNotEmpty(b, "Get Empty Tensor");
    const auto shape = BroadcastShape(a.shape(), b.shape());
    Tensor lhs = a.shape() == shape ? a : a.broadcast(shape);
    Tensor rhs = b.shape() == shape ? b : b.broadcast(shape);
    CheckSameDevice(lhs, rhs, "Tensor must be same device");
    CheckSameDtype(lhs, rhs, "Tensor must be same dtype");
    CheckSameShape(lhs, rhs, "Tensor must be same size");
    Tensor out(shape, a.dtype(), a.device());
    if (a.device() == Device::kCPU) {
        DivCPU(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), a.dtype(),
               shape, lhs.stride(), rhs.stride(), out.stride());
    } else if (a.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        DivCUDA(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel(), a.dtype(),
                out.shape_cuda(), lhs.stride_cuda(), rhs.stride_cuda(), out.dim());
#else
        throw std::logic_error("div: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("div: unsupported device");
    }

    if (EnableGrad() && CheckGradDtype(a.dtype())) {
        auto node = std::make_shared<DivNode>();
        node->inputs = { lhs, rhs };
        node->input_versions[0] = lhs.version();
        node->input_versions[1] = rhs.version();
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor neg(const Tensor& a){
    CheckNotEmpty(a, "Get Empty Tensor");
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

    if (EnableGrad() && CheckGradDtype(a.dtype())) {
        auto node = std::make_shared<NegNode>();
        node->inputs = { a };
        out.set_grad_node(std::move(node));
    }

    return out;
}

}  // namespace mtensor
