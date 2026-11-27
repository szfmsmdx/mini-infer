#include <memory>
#include <utility>

#include "mini_infer/autograd.h"
#include "mini_infer/ops/elementwise.h"
#include "tensor/tensor_impl.h"

namespace mtensor {

AutogradMetadata& Tensor::ensure_autograd_metadata() {
    if (!impl_) impl_ = EmptyWithOptions(dtype(), device()).impl_;
    if (!impl_->autograd_meta) impl_->autograd_meta = std::make_shared<AutogradMetadata>();
    return *impl_->autograd_meta;
}

bool Tensor::requires_grad() const {
    return impl_ && impl_->autograd_meta && impl_->autograd_meta->requires_grad;
}

Tensor& Tensor::requires_grad_(bool value) {
    // TODO: 实现反向传播时确定各 dtype 的梯度支持范围及检查位置。
    ensure_autograd_metadata().requires_grad = value;
    return *this;
}

bool Tensor::has_grad() const {
    return impl_ && impl_->autograd_meta && impl_->autograd_meta->grad.has_value();
}

Tensor Tensor::grad() const {
    return has_grad() ? *impl_->autograd_meta->grad : Tensor();
}

void Tensor::zero_grad() {
    if(has_grad()){
        impl_->autograd_meta->grad->zero_();
    }
}

std::shared_ptr<AutogradNode> Tensor::grad_node() const {
    return impl_ && impl_->autograd_meta ? impl_->autograd_meta->grad_node : nullptr;
}

void Tensor::set_grad_node(std::shared_ptr<AutogradNode> node) {
    ensure_autograd_metadata().grad_node = std::move(node);
}

void Tensor::accumulate_grad(const Tensor& grad) {
    // 从零开始累计梯度，如果没有的话就保存
    auto& meta = ensure_autograd_metadata();
    if (!meta.grad.has_value()) {
        meta.grad = grad.clone();
    } else {
        *meta.grad = add(*meta.grad, grad);
    }
}

Tensor Tensor::detach() const {
    if (!impl_) return EmptyWithOptions(dtype(), device());
    return MakeView(shape(), stride());
}

void Tensor::backward() { RunBackward(*this); }
void Tensor::backward(const Tensor& gradient) { RunBackward(*this, gradient); }

}  // namespace mtensor
