#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>

#include "mini_infer/ops/elementwise.h"
#include "ops/elementwise/kernels.h"
#include "ops/common/cpu_iteration.h"
#include "tensor/check.h"
#include "mini_infer/autograd.h"

namespace mtensor {

class AddNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SubNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class NegNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SqrtNode : public AutogradNode {
public:
    std::uint64_t input_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class RsqrtNode : public AutogradNode {
public:
    std::uint64_t input_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class ExpNode : public AutogradNode {
public:
    Tensor y;
    std::uint64_t output_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class LogNode : public AutogradNode {
public:
    std::uint64_t input_version = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class MulNode : public AutogradNode {
public:
    std::uint64_t input_versions[2] = {};
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class DivNode : public AutogradNode {
public:
    std::uint64_t input_versions[2] = {};
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

namespace {

Tensor ScalarLike(const Scalar& value, const Tensor& x) {
    return full({}, value, x.dtype(), x.device());
}

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

std::pair<Tensor, Tensor> PrepareBinaryInputs(const Tensor& a, const Tensor& b) {
    CheckNotEmpty(a, "Get Empty Tensor");
    CheckNotEmpty(b, "Get Empty Tensor");
    const auto shape = BroadcastShape(a.shape(), b.shape());
    Tensor lhs = a.shape() == shape ? a : a.broadcast(shape);
    Tensor rhs = b.shape() == shape ? b : b.broadcast(shape);
    CheckSameDevice(lhs, rhs, "Tensor must be same device");
    CheckSameDtype(lhs, rhs, "Tensor must be same dtype");
    return {lhs, rhs};
}

}  // namespace


Tensor add(const Tensor& a, const Tensor& b){
    auto [lhs, rhs] = PrepareBinaryInputs(a, b);
    const auto& shape = lhs.shape();
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
    auto [lhs, rhs] = PrepareBinaryInputs(a, b);
    const auto& shape = lhs.shape();
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
    auto [lhs, rhs] = PrepareBinaryInputs(a, b);
    const auto& shape = lhs.shape();
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
    auto [lhs, rhs] = PrepareBinaryInputs(a, b);
    const auto& shape = lhs.shape();
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

Tensor sqrt(const Tensor& x) {
    CheckNotEmpty(x, "sqrt: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor out(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    if (x.device() == Device::kCPU) {
        SqrtCPU(x.data_ptr(), out.data_ptr(), x.dtype(), x.shape(), x.stride());
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        SqrtCUDA(x.data_ptr(), out.data_ptr(), x.dtype(), x.numel(),
                 x.shape_cuda(), x.stride_cuda(), x.dim());
#endif
    } else {
        throw std::logic_error("sqrt: no backend for this device");
    }
    if (EnableGrad() && floating) {
        auto node = std::make_shared<SqrtNode>();
        node->inputs = {x};
        node->input_version = x.version();
        out.set_grad_node(std::move(node));
    }
    return out;
}

Tensor rsqrt(const Tensor& x) {
    CheckNotEmpty(x, "rsqrt: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor out(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    if (x.device() == Device::kCPU) {
        RsqrtCPU(x, out);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        RsqrtCUDA(x, out);
#endif
    } else {
        throw std::logic_error("rsqrt: no backend for this device");
    }
    if (EnableGrad() && floating) {
        auto node = std::make_shared<RsqrtNode>();
        node->inputs = {x};
        node->input_version = x.version();
        out.set_grad_node(std::move(node));
    }
    return out;
}

Tensor exp(const Tensor& x) {
    CheckNotEmpty(x, "exp: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor out(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    if (x.device() == Device::kCPU) {
        ExpCPU(x, out);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        ExpCUDA(x, out);
#endif
    } else {
        throw std::logic_error("exp: no backend for this device");
    }
    if (EnableGrad() && floating) {
        auto node = std::make_shared<ExpNode>();
        node->inputs = {x};
        node->y = out.detach();
        node->output_version = out.version();
        out.set_grad_node(std::move(node));
    }
    return out;
}

Tensor log(const Tensor& x) {
    CheckNotEmpty(x, "log: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor out(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    if (x.device() == Device::kCPU) {
        LogCPU(x, out);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        LogCUDA(x, out);
#endif
    } else {
        throw std::logic_error("log: no backend for this device");
    }
    if (EnableGrad() && floating) {
        auto node = std::make_shared<LogNode>();
        node->inputs = {x};
        node->input_version = x.version();
        out.set_grad_node(std::move(node));
    }
    return out;
}

Tensor Tensor::sqrt() const { return mtensor::sqrt(*this); }
Tensor Tensor::rsqrt() const { return mtensor::rsqrt(*this); }
Tensor Tensor::exp() const { return mtensor::exp(*this); }
Tensor Tensor::log() const { return mtensor::log(*this); }

Tensor Tensor::operator+(const Tensor& other) const { return add(*this, other); }

Tensor Tensor::operator-(const Tensor& other) const { return sub(*this, other); }

Tensor Tensor::operator*(const Tensor& other) const { return mul(*this, other); }

Tensor Tensor::operator/(const Tensor& other) const { return div(*this, other); }

Tensor Tensor::operator+(const Scalar& value) const { return add(*this, ScalarLike(value, *this)); }
Tensor Tensor::operator-(const Scalar& value) const { return sub(*this, ScalarLike(value, *this)); }
Tensor Tensor::operator*(const Scalar& value) const { return mul(*this, ScalarLike(value, *this)); }
Tensor Tensor::operator/(const Scalar& value) const { return div(*this, ScalarLike(value, *this)); }

Tensor operator+(const Scalar& value, const Tensor& x) { return x + value; }
Tensor operator-(const Scalar& value, const Tensor& x) { return sub(ScalarLike(value, x), x); }
Tensor operator*(const Scalar& value, const Tensor& x) { return x * value; }
Tensor operator/(const Scalar& value, const Tensor& x) { return div(ScalarLike(value, x), x); }

Tensor Tensor::operator-() const { return neg(*this); }

namespace {

template <typename T>
void RsqrtBackwardTypedCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto* input = static_cast<const std::byte*>(x.data_ptr());
    const auto* grad = static_cast<const std::byte*>(output_grad.data_ptr());
    auto* output = static_cast<std::byte*>(dx.data_ptr());
    ForEachOffset(x.shape(), x.stride(), output_grad.stride(),
        [&](std::size_t xi, std::size_t gi) {
            T value(0), upstream(0);
            std::memcpy(&value, input + xi * sizeof(T), sizeof(T));
            std::memcpy(&upstream, grad + gi * sizeof(T), sizeof(T));
            const float y = 1.0f / std::sqrt(static_cast<float>(value));
            // y 和乘法中间结果保持 FP32，最后才写回原 dtype。
            const T result(((-0.5f * y) * static_cast<float>(upstream)) * y * y);
            std::memcpy(output, &result, sizeof(T));
            output += sizeof(T);
        });
}

void RsqrtBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: RsqrtBackwardTypedCPU<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: RsqrtBackwardTypedCPU<Float16>(x, output_grad, dx); break;
        case Dtype::kBFloat16: RsqrtBackwardTypedCPU<BFloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("rsqrt backward: unsupported dtype");
    }
}

}  // namespace

std::vector<Tensor> AddNode::backward(const Tensor& output_grad) {
    return {output_grad, output_grad};
}

std::vector<Tensor> SubNode::backward(const Tensor& output_grad) {
    return {output_grad, neg(output_grad)};
}

std::vector<Tensor> NegNode::backward(const Tensor& output_grad) {
    return {neg(output_grad)};
}

std::vector<Tensor> SqrtNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_version, "SqrtNode::backward: input modified after forward");
    const Tensor root = sqrt(inputs[0]);
    return {output_grad / (root + root)};
}

std::vector<Tensor> RsqrtNode::backward(const Tensor& output_grad) {
    const Tensor& x = inputs[0];
    CheckVersion(x, input_version, "RsqrtNode::backward: input modified after forward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::kCPU) {
        RsqrtBackwardCPU(x, output_grad, dx);
#ifdef ENABLE_CUDA
    } else if (x.device() == Device::kCUDA) {
        RsqrtBackwardCUDA(x, output_grad, dx);
#endif
    } else {
        throw std::logic_error("rsqrt backward: no backend for this device");
    }
    return {dx};
}

std::vector<Tensor> ExpNode::backward(const Tensor& output_grad) {
    CheckVersion(y, output_version, "ExpNode::backward: output modified after forward");
    return {output_grad * y};
}

std::vector<Tensor> LogNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_version, "LogNode::backward: input modified after forward");
    return {output_grad / inputs[0]};
}

std::vector<Tensor> MulNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_versions[0], "MulNode::backward: input modified after forward");
    CheckVersion(inputs[1], input_versions[1], "MulNode::backward: input modified after forward");
    return {mul(output_grad, inputs[1]), mul(output_grad, inputs[0])};
}

std::vector<Tensor> DivNode::backward(const Tensor& output_grad) {
    CheckVersion(inputs[0], input_versions[0], "DivNode::backward: input modified after forward");
    CheckVersion(inputs[1], input_versions[1], "DivNode::backward: input modified after forward");
    return {div(output_grad, inputs[1]),
            -output_grad * ((inputs[0] / inputs[1]) / inputs[1])};
}

}  // namespace mtensor
