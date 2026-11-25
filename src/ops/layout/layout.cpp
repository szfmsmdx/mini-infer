#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/tensor/tensor.h"
#include "tensor/check.h"
#include "tensor/tensor_impl.h"
#include "mini_infer/autograd.h"

namespace mtensor {

class TransposeNode : public AutogradNode {
public:
    int dim0, dim1;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class PermuteNode : public AutogradNode {
public:
    std::vector<int> dims;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class BroadcastNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

// inputs[0] 保存前向输入，原 shape/dtype/device 从中读取。
class ReshapeNode : public AutogradNode {
public:
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class NarrowNode : public AutogradNode {
public:
    int dim = 0;
    int start = 0;
    int length = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

// operator[] 只沿第 0 维索引，并移除该维度。
class IndexNode : public AutogradNode {
public:
    int index = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class SqueezeNode : public AutogradNode {
public:
    // 保存实际移除的维度；未移除维度时不使用此节点。
    int dim = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

class UnsqueezeNode : public AutogradNode {
public:
    // 保存实际插入的维度。
    int dim = 0;
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};


namespace {
std::vector<int> InferReshapeShape(std::vector<int> shape, std::size_t numel) {
    // 检查目标维度，记录唯一的 -1，并计算已知维度的乘积。
    std::size_t count = 1;
    std::size_t inferred_dim = shape.size();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        const int dimension = shape[i];
        if (dimension == -1) {
            if (inferred_dim != shape.size()) {
                throw std::invalid_argument("Tensor::reshape: only one inferred dimension is allowed");
            }
            inferred_dim = i;
            continue;
        }
        if (dimension <= 0) {
            throw std::invalid_argument(
                "Tensor::reshape: dimension must be positive or -1");
        }
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("Tensor::reshape: shape product overflow");
        }
        count *= dimension;
    }
    // 有 -1 时根据总元素数推导该维；否则检查目标元素数是否一致。
    if (inferred_dim != shape.size()) {
        if (numel == 0 || numel % count != 0) {
            throw std::invalid_argument("Tensor::reshape: cannot infer dimension");
        }
        const std::size_t inferred_size = numel / count;
        if (inferred_size > std::numeric_limits<int>::max()) {
            throw std::overflow_error("Tensor::reshape: inferred dimension exceeds int range");
        }
        shape[inferred_dim] = static_cast<int>(inferred_size);
    } else if (count != numel) {
        throw std::invalid_argument(
            "Tensor::reshape: shape must match");
    }

    return shape;
}

std::vector<std::size_t> ContiguousStride(ShapeView shape) {
    std::vector<std::size_t> result(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i > 0; --i) {
        result[i - 1] = stride;
        stride *= static_cast<std::size_t>(shape[i - 1]);
    }
    return result;
}

}  // namespace

Tensor Tensor::MakeView(ShapeView shape, StrideView stride, std::size_t offset_delta) const {
    if (!impl_) return EmptyWithOptions(dtype(), device());
    const auto parent_offset = impl_->storage_offset;
    if (offset_delta > std::numeric_limits<std::size_t>::max() - parent_offset) {
        throw std::overflow_error("Tensor view: storage offset overflow");
    }
    Tensor result = *this;
    result.impl_ = std::make_shared<TensorImpl>(impl_->storage, shape, stride,
        parent_offset + offset_delta, dtype(), device(), impl_->version);
    return result;
}

Tensor Tensor::operator[](int index) const {
    if(this->shape().empty() || index < 0 || index >= this->shape()[0]){
        throw std::out_of_range("Tensor::operator[]: index out of range");
    }
    CheckDataPointer(*this, "Tensor::operator[]: null data pointer");

    const std::size_t offset = static_cast<std::size_t>(index) * this->stride()[0];
    std::vector<int> shape(this->shape().begin() + 1, this->shape().end());
    std::vector<std::size_t> stride(this->stride().begin() + 1, this->stride().end());
    Tensor result = MakeView(shape, stride, offset);
    if (EnableGrad() && CheckGradDtype(dtype())) {
        auto node = std::make_shared<IndexNode>();
        node->inputs = {*this};
        node->index = index;
        result.set_grad_node(std::move(node));
    }

    return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    /*
    使用示例:
        Tensor b = a.transpose(0, 1);
    */
    CheckDim(dim0, this->shape().size(), "Tensor::transpose: dimension index out of range");
    CheckDim(dim1, this->shape().size(), "Tensor::transpose: dimension index out of range");

    auto shape = this->shape().to_vector();
    auto stride = this->stride().to_vector();
    shape[dim0] = this->shape()[dim1];
    shape[dim1] = this->shape()[dim0];
    stride[dim0] = this->stride()[dim1];
    stride[dim1] = this->stride()[dim0];
    Tensor result = MakeView(shape, stride);

    if(EnableGrad() && CheckGradDtype(dtype())) {
        auto node = std::make_shared<TransposeNode>();
        node->inputs = { *this };
        node->dim0 = dim0;
        node->dim1 = dim1;
        result.set_grad_node(node);
    }
    return result;
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    /*
    使用示例:
        Tensor b = a.permute({2, 0, 1});
    */

    // 直接交换 stride 和 shape 即可
    // 新旧下标取到同一个元素的 offset 应该一致，那么一个简单的直觉证明就是直接交换
    // offset(b) = Σ jₖ × b.stride[k] = Σ jₖ × a.stride[dims[k]]

    CheckPermutation(dims, this->shape().size());

    auto shape = this->shape().to_vector();
    auto stride = this->stride().to_vector();
    for (std::size_t i = 0; i < dims.size(); ++i) {
        shape[i] = this->shape()[dims[i]];
        stride[i] = this->stride()[dims[i]];
    }
    Tensor result = MakeView(shape, stride);

    if(EnableGrad() && CheckGradDtype(dtype())){
        auto node = std::make_shared<PermuteNode>();
        node->inputs = { *this };
        node->dims = dims;
        result.set_grad_node(node);
    }

    return result;
}

Tensor Tensor::contiguous() const {
    /*
    使用示例:
        Tensor b = a.contiguous();
        tensor b 在不连续的情况下申请一块薪内存，存储新的数据分布
    */

    if(is_contiguous())
        return *this;

    return clone(); // 反向传播也借用 clone 了
}

Tensor Tensor::squeeze(int dim) const {
    /*
    使用示例:
        Tensor b = a.squeeze(1);
    */
    CheckNotEmpty(*this, "Tensor::squeeze: input must not be empty");
    CheckDim(dim, this->shape().size(), "Tensor::squeeze: dimension index out of range");
    if(this->shape()[dim] != 1)
        return *this;

    auto shape = this->shape().to_vector();
    auto stride = this->stride().to_vector();
    shape.erase(shape.begin() + dim);
    stride.erase(stride.begin() + dim);
    Tensor res = MakeView(shape, stride);

    if(EnableGrad() && CheckGradDtype(dtype())){
        auto node = std::make_shared<SqueezeNode>();
        node->inputs = { *this };
        node->dim = dim;
        res.set_grad_node(node);
    }

    return res;
}

Tensor Tensor::unsqueeze(int dim) const {
    /*
    使用示例:
        Tensor b = a.unsqueeze(1);
    */
    CheckNotEmpty(*this, "Tensor::unsqueeze: input must not be empty");
    CheckDim(dim, this->shape().size() + 1, "Tensor::unsqueeze: dimension index out of range");
    CheckLayoutRank(this->shape().size() + 1);
    auto shape = this->shape().to_vector();
    auto stride = this->stride().to_vector();
    std::size_t inserted_stride = 1;
    if (dim != this->dim()) {
        const auto extent = static_cast<std::size_t>(this->shape()[dim]);
        if (this->stride()[dim] > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error("Tensor::unsqueeze: stride overflow");
        }
        inserted_stride = this->stride()[dim] * extent;
    }
    shape.insert(shape.begin() + dim, 1);
    stride.insert(stride.begin() + dim, inserted_stride);
    Tensor res = MakeView(shape, stride);

    if(EnableGrad() && CheckGradDtype(dtype())){
        auto node = std::make_shared<UnsqueezeNode>();
        node->dim = dim;
        node->inputs = { *this };
        res.set_grad_node(node);
    }

    return res;
}

Tensor Tensor::reshape(ShapeView requested_shape) const {
    CheckLayoutRank(requested_shape.size());
    auto shape = InferReshapeShape(requested_shape.to_vector(), numel());

    // 非连续输入先按逻辑顺序复制，复用 CloneNode 和 ReshapeNode 的反向链。
    if (!is_contiguous()) {
        return clone().reshape(std::move(shape));
    }

    // 连续输入创建共享存储的新视图，更新布局并使用独立的 Autograd 元数据。
    auto stride = ContiguousStride(shape);
    Tensor result = MakeView(shape, stride);

    // 保存前向输入，反向时将梯度恢复为输入 shape。
    if(EnableGrad() && CheckGradDtype(dtype())){
        auto node = std::make_shared<ReshapeNode>();
        node->inputs = { *this };
        result.set_grad_node(node);
    }

    return result;
}

Tensor Tensor::narrow(int dim, int start, int length) const{
    CheckDim(dim, this->shape().size(), "Tensor::narrow: dimension index out of range");
    const int dimension = this->shape()[dim];
    if (start < 0 || start >= dimension) {
        throw std::out_of_range("Tensor::narrow: start index out of range");
    }
    if (length <= 0 || length > dimension - start) {
        throw std::invalid_argument("Tensor::narrow: length out of range");
    }
    CheckDataPointer(*this, "Tensor::narrow: null data pointer");

    auto shape = this->shape().to_vector();
    shape[dim] = length;

    const std::size_t offset_delta = static_cast<std::size_t>(start) * this->stride()[dim];
    Tensor result = MakeView(shape, this->stride(), offset_delta);
    if (EnableGrad() && CheckGradDtype(dtype())) {
        auto node = std::make_shared<NarrowNode>();
        node->inputs = {*this};
        node->dim = dim;
        node->start = start;
        node->length = length;
        result.set_grad_node(std::move(node));
    }
    return result;
}

Tensor Tensor::broadcast(ShapeView target_shape) const {
    CheckNotEmpty(*this, "broadcast: input must not be empty");
    CheckLayoutRank(target_shape.size());
    CheckBroadcastTarget(this->shape(), target_shape);
    const std::size_t leading = target_shape.size() - this->shape().size();

    std::vector<std::size_t> stride(target_shape.size(), 0);
    for (std::size_t i = 0; i < this->shape().size(); ++i) {
        if (this->shape()[i] != 1 || target_shape[leading + i] == 1) {
            stride[leading + i] = this->stride()[i];
        }
    }
    Tensor result = MakeView(target_shape, stride);
    if (EnableGrad() && CheckGradDtype(dtype())) {
        auto node = std::make_shared<BroadcastNode>();
        node->inputs = { *this };
        result.set_grad_node(node);
    }

    return result;
}


std::vector<Tensor> TransposeNode::backward(const Tensor& output_grad) {
    return { output_grad.transpose(dim0, dim1) };
}

std::vector<Tensor> PermuteNode::backward(const Tensor& output_grad) {
    std::vector<int> origin_dims = dims;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        origin_dims[dims[i]] = i;
    }

    return { output_grad.permute(origin_dims) };
}

std::vector<Tensor> BroadcastNode::backward(const Tensor& output_grad) {
    Tensor res = output_grad;
    const auto& input_shape = inputs[0].shape();
    const int leading = output_grad.dim() - inputs[0].dim();
    for (int i = output_grad.dim() - 1; i >= 0; --i) {
        if(i >= leading && input_shape[i - leading] == 1){
            res = res.sum(i, true);
        }
        if(i < leading)
            res = res.sum(i, false);
    }
    return { res };
}

std::vector<Tensor> ReshapeNode::backward(const Tensor& output_grad) {
    auto contiguous_grad = output_grad.is_contiguous() ? output_grad : output_grad.clone();
    return { contiguous_grad.reshape(inputs[0].shape()) };
}

std::vector<Tensor> NarrowNode::backward(const Tensor& output_grad) {
    Tensor res(inputs[0].shape(), inputs[0].dtype(), inputs[0].device());
    res.zero_();
    res.narrow(dim, start, length).copy_(output_grad);
    return { res };
}

std::vector<Tensor> IndexNode::backward(const Tensor& output_grad) {
    Tensor res(inputs[0].shape(), inputs[0].dtype(), inputs[0].device());
    res.zero_();
    res[index].copy_(output_grad);
    return { res };
}

std::vector<Tensor> SqueezeNode::backward(const Tensor& output_grad) {
    return { output_grad.unsqueeze(dim) };
}

std::vector<Tensor> UnsqueezeNode::backward(const Tensor& output_grad) {
    return {output_grad.squeeze(dim)};
}

}  // namespace mtensor
