#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/tensor.h"
#include "tensor/check.h"
#include "mini_infer/autograd.h"
#include "autograd/backward_nodes/tensor_ops.h"

namespace mtensor {

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

Tensor Tensor::operator[](int index) const {
    if(shape_.empty() || index < 0 || index >= size(0)){
        throw std::out_of_range("Tensor::operator[]: index out of range");
    }
    CheckDataPointer(*this, "Tensor::operator[]: null data pointer");

    const std::size_t offset = static_cast<std::size_t>(index) * this->stride_[0] * element_size();
    Tensor result;
    result.device_ = device_, result.dtype_ = dtype_;
    result.shape_.assign(shape_.begin() + 1, shape_.end());
    result.stride_.assign(stride_.begin() + 1, stride_.end());
    result.data_ptr_ = static_cast<std::byte*>(data_ptr_) + offset;
    result.storage_ = storage_;
    result.version_ = version_;
    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }
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

    Tensor result(*this);
    result.autograd_metadata_ = std::make_shared<AutogradMetadata>();
    result.shape_[dim0] = shape_[dim1];
    result.shape_[dim1] = shape_[dim0];
    result.stride_[dim0] = stride_[dim1];
    result.stride_[dim1] = stride_[dim0];

    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }

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

    Tensor result = *this;
    result.autograd_metadata_ = std::make_shared<AutogradMetadata>();
    for (std::size_t i = 0; i < dims.size(); ++i) {
        result.shape_[i] = shape_[dims[i]];
        result.stride_[i] = stride_[dims[i]];
    }

    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }

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

    Tensor res(*this);
    res.autograd_metadata_ = std::make_shared<AutogradMetadata>();
    res.shape_.erase(res.shape_.begin() + dim);
    res.stride_.erase(res.stride_.begin() + dim);

    if(device_ == Device::kCUDA) {
        res.InitCUDAMetadata();
    }

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
    Tensor res(*this);
    res.autograd_metadata_ = std::make_shared<AutogradMetadata>();
    const std::size_t inserted_stride =
        dim == this->dim() ? 1 : stride_[dim] * shape_[dim];
    res.shape_.insert(res.shape_.begin() + dim, 1);
    res.stride_.insert(res.stride_.begin() + dim, inserted_stride);
    if(device_ == Device::kCUDA){
        res.InitCUDAMetadata();
    }

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
    Tensor result = *this;
    result.shape_ = std::move(shape);
    result.stride_ = ContiguousStride(result.shape_);
    result.autograd_metadata_ = std::make_shared<AutogradMetadata>();
    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }

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
    const int dimension = size(static_cast<std::size_t>(dim));
    if (start < 0 || start >= dimension) {
        throw std::out_of_range("Tensor::narrow: start index out of range");
    }
    if (length <= 0 || length > dimension - start) {
        throw std::invalid_argument("Tensor::narrow: length out of range");
    }
    CheckDataPointer(*this, "Tensor::narrow: null data pointer");

    Tensor result;
    result.storage_ = storage_;
    result.version_ = version_;
    result.shape_ = shape_;
    result.shape_[dim] = length;
    result.stride_ = stride_;
    result.dtype_ = dtype_;
    result.device_ = device_;

    const std::size_t byte_offset = static_cast<std::size_t>(start) * stride_[dim] * element_size();
    result.data_ptr_ = static_cast<std::byte*>(data_ptr_) + byte_offset;
    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }
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

    Tensor result = *this;
    result.autograd_metadata_ = std::make_shared<AutogradMetadata>();
    result.shape_ = target_shape;
    result.stride_.assign(target_shape.size(), 0);
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != 1 || target_shape[leading + i] == 1) {
            result.stride_[leading + i] = stride_[i];
        }
    }
    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }
    if (EnableGrad() && CheckGradDtype(dtype_)) {
        auto node = std::make_shared<BroadcastNode>();
        node->inputs = { *this };
        result.set_grad_node(node);
    }

    return result;
}

}  // namespace mtensor
