#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/tensor.h"
#include "tensor/check.h"
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

}  // namespace

std::vector<std::size_t> Tensor::ContiguousStride(const std::vector<int>& shape) {
    std::vector<std::size_t> result(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i > 0; --i) {
        result[i - 1] = stride;
        stride *= static_cast<std::size_t>(shape[i - 1]);
    }
    return result;
}

Tensor Tensor::MakeView(std::vector<int> shape, std::vector<std::size_t> stride,
                        std::size_t byte_offset) const {
    Tensor result(*this);
    result.autograd_metadata_ = std::make_shared<AutogradMetadata>();
    result.shape_ = std::move(shape);
    result.stride_ = std::move(stride);
    if (byte_offset != 0) {
        result.data_ptr_ = static_cast<std::byte*>(data_ptr_) + byte_offset;
    }
    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }
    return result;
}

Tensor Tensor::operator[](int index) const {
    if(shape_.empty() || index < 0 || index >= shape_[0]){
        throw std::out_of_range("Tensor::operator[]: index out of range");
    }
    CheckDataPointer(*this, "Tensor::operator[]: null data pointer");

    const std::size_t offset = static_cast<std::size_t>(index) * this->stride_[0] * element_size();
    std::vector<int> shape(shape_.begin() + 1, shape_.end());
    std::vector<std::size_t> stride(stride_.begin() + 1, stride_.end());
    Tensor result = MakeView(std::move(shape), std::move(stride), offset);
    if (EnableGrad() && CheckGradDtype(dtype_)) {
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
    CheckDim(dim0, shape_.size(), "Tensor::transpose: dimension index out of range");
    CheckDim(dim1, shape_.size(), "Tensor::transpose: dimension index out of range");

    auto shape = shape_;
    auto stride = stride_;
    shape[dim0] = shape_[dim1];
    shape[dim1] = shape_[dim0];
    stride[dim0] = stride_[dim1];
    stride[dim1] = stride_[dim0];
    Tensor result = MakeView(std::move(shape), std::move(stride));

    if(EnableGrad() && CheckGradDtype(dtype_)) {
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

    CheckPermutation(dims, shape_.size());

    auto shape = shape_;
    auto stride = stride_;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        shape[i] = shape_[dims[i]];
        stride[i] = stride_[dims[i]];
    }
    Tensor result = MakeView(std::move(shape), std::move(stride));

    if(EnableGrad() && CheckGradDtype(dtype_)){
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
    CheckDim(dim, shape_.size(), "Tensor::squeeze: dimension index out of range");
    if(shape_[dim] != 1)
        return *this;

    auto shape = shape_;
    auto stride = stride_;
    shape.erase(shape.begin() + dim);
    stride.erase(stride.begin() + dim);
    Tensor res = MakeView(std::move(shape), std::move(stride));

    if(EnableGrad() && CheckGradDtype(dtype_)){
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
    CheckDim(dim, shape_.size() + 1, "Tensor::unsqueeze: dimension index out of range");
    auto shape = shape_;
    auto stride = stride_;
    const std::size_t inserted_stride =
        dim == this->dim() ? 1 : stride_[dim] * shape_[dim];
    shape.insert(shape.begin() + dim, 1);
    stride.insert(stride.begin() + dim, inserted_stride);
    Tensor res = MakeView(std::move(shape), std::move(stride));

    if(EnableGrad() && CheckGradDtype(dtype_)){
        auto node = std::make_shared<UnsqueezeNode>();
        node->dim = dim;
        node->inputs = { *this };
        res.set_grad_node(node);
    }

    return res;
}

Tensor Tensor::reshape(std::vector<int> shape) const {
    shape = InferReshapeShape(std::move(shape), numel());

    // 非连续输入先按逻辑顺序复制，复用 CloneNode 和 ReshapeNode 的反向链。
    if (!is_contiguous()) {
        return clone().reshape(std::move(shape));
    }

    // 连续输入创建共享存储的新视图，更新布局并使用独立的 Autograd 元数据。
    auto stride = ContiguousStride(shape);
    Tensor result = MakeView(std::move(shape), std::move(stride));

    // 保存前向输入，反向时将梯度恢复为输入 shape。
    if(EnableGrad() && CheckGradDtype(dtype_)){
        auto node = std::make_shared<ReshapeNode>();
        node->inputs = { *this };
        result.set_grad_node(node);
    }

    return result;
}

Tensor Tensor::narrow(int dim, int start, int length) const{
    CheckDim(dim, shape_.size(), "Tensor::narrow: dimension index out of range");
    const int dimension = shape_[dim];
    if (start < 0 || start >= dimension) {
        throw std::out_of_range("Tensor::narrow: start index out of range");
    }
    if (length <= 0 || length > dimension - start) {
        throw std::invalid_argument("Tensor::narrow: length out of range");
    }
    CheckDataPointer(*this, "Tensor::narrow: null data pointer");

    auto shape = shape_;
    shape[dim] = length;

    const std::size_t byte_offset = static_cast<std::size_t>(start) * stride_[dim] * element_size();
    Tensor result = MakeView(std::move(shape), stride_, byte_offset);
    if (EnableGrad() && CheckGradDtype(dtype_)) {
        auto node = std::make_shared<NarrowNode>();
        node->inputs = {*this};
        node->dim = dim;
        node->start = start;
        node->length = length;
        result.set_grad_node(std::move(node));
    }
    return result;
}

Tensor Tensor::broadcast(const std::vector<int>& target_shape) const {
    CheckNotEmpty(*this, "broadcast: input must not be empty");
    CheckBroadcastTarget(shape_, target_shape);
    const std::size_t leading = target_shape.size() - shape_.size();

    std::vector<std::size_t> stride(target_shape.size(), 0);
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != 1 || target_shape[leading + i] == 1) {
            stride[leading + i] = stride_[i];
        }
    }
    Tensor result = MakeView(target_shape, std::move(stride));
    if (EnableGrad() && CheckGradDtype(dtype_)) {
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
