#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/ops/elementwise.h"
#include "backend/dispatch.h"
#include "mini_infer/autograd.h"
#include "tensor/check.h"

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
    SavedTensor saved_input;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class RsqrtNode : public AutogradNode {
public:
    SavedTensor saved_input;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class ExpNode : public AutogradNode {
public:
    SavedTensor saved_output;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class LogNode : public AutogradNode {
public:
    SavedTensor saved_input;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class MulNode : public AutogradNode {
public:
    SavedTensor saved_inputs[2];
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class DivNode : public AutogradNode {
public:
    SavedTensor saved_inputs[2];
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

namespace {

Tensor ScalarLike(const Scalar& value, const Tensor& x) {
    return full({}, value, x.dtype(), x.device());
}

std::vector<int> BroadcastShape(ShapeView a, ShapeView b) {
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

void CheckOutStorage(const Tensor& x, const Tensor& out, const char* name) {
    CheckNotEmpty(out, name);
    CheckSameShape(x, out, name);
    CheckSameDevice(x, out, name);
    if (!out.is_contiguous() || out.shares_storage(x)) {
        throw std::invalid_argument("op_out: output must be contiguous and have independent storage");
    }
}

Tensor& RunBinaryOut(const Tensor& a, const Tensor& b, Tensor& out,
                     const KernelSlot<BinaryKernel>& slot, const char* name) {
    CheckNotEmpty(a, name);
    CheckNotEmpty(b, name);
    CheckSameShape(a, b, name);
    CheckSameDtype(a, b, name);
    CheckSameDevice(a, b, name);
    CheckOutStorage(a, out, name);
    CheckSameDtype(a, out, name);
    if (out.shares_storage(b)) {
        throw std::invalid_argument("op_out: output must have independent storage");
    }
    out.update_version();
    slot.call(a.device(), name, a, b, out);
    return out;
}

Tensor& RunUnaryOut(const Tensor& x, Tensor& out, Dtype expected_dtype,
                    const KernelSlot<UnaryKernel>& slot, const char* name) {
    CheckNotEmpty(x, name);
    CheckOutStorage(x, out, name);
    if (out.dtype() != expected_dtype) {
        throw std::invalid_argument("op_out: output dtype does not match");
    }
    out.update_version();
    slot.call(x.device(), name, x, out);
    return out;
}

}  // namespace

Tensor& add_out(const Tensor& a, const Tensor& b, Tensor& out) {
    return RunBinaryOut(a, b, out, GetKernelTable().add, "add_out");
}
Tensor& sub_out(const Tensor& a, const Tensor& b, Tensor& out) {
    return RunBinaryOut(a, b, out, GetKernelTable().sub, "sub_out");
}
Tensor& mul_out(const Tensor& a, const Tensor& b, Tensor& out) {
    return RunBinaryOut(a, b, out, GetKernelTable().mul, "mul_out");
}
Tensor& div_out(const Tensor& a, const Tensor& b, Tensor& out) {
    return RunBinaryOut(a, b, out, GetKernelTable().div, "div_out");
}
Tensor& neg_out(const Tensor& x, Tensor& out) {
    return RunUnaryOut(x, out, x.dtype(), GetKernelTable().neg, "neg_out");
}
Tensor& sqrt_out(const Tensor& x, Tensor& out) {
    const Dtype dtype = CheckGradDtype(x.dtype()) ? x.dtype() : Dtype::kFloat32;
    return RunUnaryOut(x, out, dtype, GetKernelTable().sqrt, "sqrt_out");
}
Tensor& rsqrt_out(const Tensor& x, Tensor& out) {
    const Dtype dtype = CheckGradDtype(x.dtype()) ? x.dtype() : Dtype::kFloat32;
    return RunUnaryOut(x, out, dtype, GetKernelTable().rsqrt, "rsqrt_out");
}
Tensor& exp_out(const Tensor& x, Tensor& out) {
    const Dtype dtype = CheckGradDtype(x.dtype()) ? x.dtype() : Dtype::kFloat32;
    return RunUnaryOut(x, out, dtype, GetKernelTable().exp, "exp_out");
}
Tensor& log_out(const Tensor& x, Tensor& out) {
    const Dtype dtype = CheckGradDtype(x.dtype()) ? x.dtype() : Dtype::kFloat32;
    return RunUnaryOut(x, out, dtype, GetKernelTable().log, "log_out");
}


Tensor add(const Tensor& a, const Tensor& b){
    auto [lhs, rhs] = PrepareBinaryInputs(a, b);
    const auto& shape = lhs.shape();
    Tensor out(shape, a.dtype(), a.device());
    add_out(lhs, rhs, out);

    // 反向传播建图
    if (ShouldRecordGrad({&lhs, &rhs}, out.dtype())) {
        auto node = std::make_shared<AddNode>();
        node->record_inputs({lhs, rhs});
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor sub(const Tensor& a, const Tensor& b){
    auto [lhs, rhs] = PrepareBinaryInputs(a, b);
    const auto& shape = lhs.shape();
    Tensor out(shape, a.dtype(), a.device());
    sub_out(lhs, rhs, out);

    if (ShouldRecordGrad({&lhs, &rhs}, out.dtype())) {
        auto node = std::make_shared<SubNode>();
        node->record_inputs({lhs, rhs});
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor mul(const Tensor& a, const Tensor& b){
    auto [lhs, rhs] = PrepareBinaryInputs(a, b);
    const auto& shape = lhs.shape();
    Tensor out(shape, a.dtype(), a.device());
    mul_out(lhs, rhs, out);

    if (ShouldRecordGrad({&lhs, &rhs}, out.dtype())) {
        auto node = std::make_shared<MulNode>();
        node->record_inputs({lhs, rhs});
        node->saved_inputs[0] = SavedTensor(lhs);
        node->saved_inputs[1] = SavedTensor(rhs);
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor div(const Tensor& a, const Tensor& b){
    auto [lhs, rhs] = PrepareBinaryInputs(a, b);
    const auto& shape = lhs.shape();
    Tensor out(shape, a.dtype(), a.device());
    div_out(lhs, rhs, out);

    if (ShouldRecordGrad({&lhs, &rhs}, out.dtype())) {
        auto node = std::make_shared<DivNode>();
        node->record_inputs({lhs, rhs});
        node->saved_inputs[0] = SavedTensor(lhs);
        node->saved_inputs[1] = SavedTensor(rhs);
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor neg(const Tensor& a){
    CheckNotEmpty(a, "Get Empty Tensor");
    Tensor out(a.shape(), a.dtype(), a.device());
    neg_out(a, out);

    if (ShouldRecordGrad({&a}, out.dtype())) {
        auto node = std::make_shared<NegNode>();
        node->record_inputs({a});
        out.set_grad_node(std::move(node));
    }

    return out;
}

Tensor sqrt(const Tensor& x) {
    CheckNotEmpty(x, "sqrt: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor out(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    sqrt_out(x, out);
    if (floating && ShouldRecordGrad({&x}, out.dtype())) {
        auto node = std::make_shared<SqrtNode>();
        node->record_inputs({x});
        node->saved_input = SavedTensor(x);
        out.set_grad_node(std::move(node));
    }
    return out;
}

Tensor rsqrt(const Tensor& x) {
    CheckNotEmpty(x, "rsqrt: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor out(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    rsqrt_out(x, out);
    if (floating && ShouldRecordGrad({&x}, out.dtype())) {
        auto node = std::make_shared<RsqrtNode>();
        node->record_inputs({x});
        node->saved_input = SavedTensor(x);
        out.set_grad_node(std::move(node));
    }
    return out;
}

Tensor exp(const Tensor& x) {
    CheckNotEmpty(x, "exp: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor out(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    exp_out(x, out);
    if (floating && ShouldRecordGrad({&x}, out.dtype())) {
        auto node = std::make_shared<ExpNode>();
        node->record_inputs({x});
        node->saved_output = SavedTensor(out, /*is_output=*/true);
        out.set_grad_node(std::move(node));
    }
    return out;
}

Tensor log(const Tensor& x) {
    CheckNotEmpty(x, "log: empty input");
    const bool floating = CheckGradDtype(x.dtype());
    Tensor out(x.shape(), floating ? x.dtype() : Dtype::kFloat32, x.device());
    log_out(x, out);
    if (floating && ShouldRecordGrad({&x}, out.dtype())) {
        auto node = std::make_shared<LogNode>();
        node->record_inputs({x});
        node->saved_input = SavedTensor(x);
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

std::vector<Tensor> AddNode::backward(const Tensor& output_grad) {
    return {output_grad, output_grad};
}

std::vector<Tensor> SubNode::backward(const Tensor& output_grad) {
    Tensor db(output_grad.shape(), output_grad.dtype(), output_grad.device());
    GetKernelTable().sub_backward.call(
        output_grad.device(), "sub backward", output_grad, db);
    return {output_grad, db};
}

std::vector<Tensor> NegNode::backward(const Tensor& output_grad) {
    Tensor dx(output_grad.shape(), output_grad.dtype(), output_grad.device());
    GetKernelTable().neg_backward.call(
        output_grad.device(), "neg backward", output_grad, dx);
    return {dx};
}

std::vector<Tensor> SqrtNode::backward(const Tensor& output_grad) {
    const Tensor x = saved_input.unpack("SqrtNode::backward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    GetKernelTable().sqrt_backward.call(
        x.device(), "sqrt backward", x, output_grad, dx);
    return {dx};
}

std::vector<Tensor> RsqrtNode::backward(const Tensor& output_grad) {
    const Tensor x = saved_input.unpack("RsqrtNode::backward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    GetKernelTable().rsqrt_backward.call(
        x.device(), "rsqrt backward", x, output_grad, dx);
    return {dx};
}

std::vector<Tensor> ExpNode::backward(const Tensor& output_grad) {
    const Tensor y = saved_output.unpack("ExpNode::backward");
    Tensor dx(y.shape(), y.dtype(), y.device());
    GetKernelTable().exp_backward.call(
        y.device(), "exp backward", y, output_grad, dx);
    return {dx};
}

std::vector<Tensor> LogNode::backward(const Tensor& output_grad) {
    const Tensor x = saved_input.unpack("LogNode::backward");
    Tensor dx(x.shape(), x.dtype(), x.device());
    GetKernelTable().log_backward.call(
        x.device(), "log backward", x, output_grad, dx);
    return {dx};
}

std::vector<Tensor> MulNode::backward(const Tensor& output_grad) {
    const Tensor a = saved_inputs[0].unpack("MulNode::backward");
    const Tensor b = saved_inputs[1].unpack("MulNode::backward");
    Tensor da(a.shape(), a.dtype(), a.device());
    Tensor db(b.shape(), b.dtype(), b.device());
    GetKernelTable().mul_backward.call(
        a.device(), "mul backward", a, b, output_grad, da, db);
    return {da, db};
}

std::vector<Tensor> DivNode::backward(const Tensor& output_grad) {
    const Tensor a = saved_inputs[0].unpack("DivNode::backward");
    const Tensor b = saved_inputs[1].unpack("DivNode::backward");
    Tensor da(a.shape(), a.dtype(), a.device());
    Tensor db(b.shape(), b.dtype(), b.device());
    GetKernelTable().div_backward.call(
        a.device(), "div backward", a, b, output_grad, da, db);
    return {da, db};
}

}  // namespace mtensor
