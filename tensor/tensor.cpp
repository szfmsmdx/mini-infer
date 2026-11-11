#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/tensor.h"
#include "tensor/dtype.h"
#include "ops/elementwise.h"
#include "utils/memory.h"
#include "ops/backend.h"
#include "autograd/autograd.h"

namespace mtensor{

void Tensor::InitCUDAMetadata() {
#ifdef ENABLE_CUDA
    if (shape_.empty()) {
        shape_cuda_.reset();
        stride_cuda_.reset();
        return;
    }
    const std::size_t shape_bytes = shape_.size() * sizeof(int);
    const std::size_t stride_bytes = stride_.size() * sizeof(std::size_t);
    std::shared_ptr<int> shape_cuda(static_cast<int*>(AllocateCUDA(shape_bytes)), FreeCUDA);
    std::shared_ptr<std::size_t> stride_cuda(
        static_cast<std::size_t*>(AllocateCUDA(stride_bytes)), FreeCUDA);
    CopyHostToCUDA(shape_cuda.get(), shape_.data(), shape_bytes);
    CopyHostToCUDA(stride_cuda.get(), stride_.data(), stride_bytes);
    shape_cuda_ = std::move(shape_cuda);
    stride_cuda_ = std::move(stride_cuda);
#else
    throw std::logic_error("Tensor: CUDA support is disabled");
#endif
}

// std::shared_ptr<AutogradMeta> 默认构造出来是空指针，不会创建 AutogradMeta
// 所以需要手动构造，而不能直接 = default
Tensor::Tensor() : autograd_metadata_(std::make_shared<AutogradMetadata>()) {}

Tensor::Tensor(const Scalar& value)
    : Tensor(/*shape=*/std::vector<int>{},
             /*dtype=*/value.is_floating_point() ? Dtype::kFloat32 : Dtype::kInt64,
             /*device=*/Device::kCPU) {
    if (value.is_unsigned_integer() &&
        value.to<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::out_of_range("Tensor: integer value is out of Int64 range");
    }
    fill_(value);
}

Tensor::Tensor(const void* data_ptr, std::vector<int> shape, Dtype dtype, Device device)
: shape_(std::move(shape)), dtype_(dtype), device_(device),
    autograd_metadata_(std::make_shared<AutogradMetadata>()) {
    /*
    使用示例：
        float values[] = {1.0f, 2.0f};
        Tensor a(values, {2}, Dtype::kFloat32);
    */
    for (int dimension : this->shape_) {
        if (dimension < 0) {
            throw std::invalid_argument("Tensor dimensions must be non-negative");
        }
    }

    stride_.resize(shape_.size());
    std::size_t stride = 1;
    for (std::size_t i = shape_.size(); i > 0; --i) {   // shape 可能为空的写法
        stride_[i - 1] = stride;
        stride *= shape_[i - 1];
    }

    // 传入数据且 shape 为 {} 时构造标量，占一个元素。
    // 此时成员数据指针尚未设置，不能靠 numel() 判断标量的大小。
    const std::size_t bytes = this->shape_.empty() ? element_size() : nbytes();
    if(bytes == 0 || data_ptr == nullptr){
        throw std::invalid_argument("Tensor constructor: null data or zero-sized shape");
    }

    if(device == Device::kCPU){
        storage_ = std::shared_ptr<void>(AllocateCPU(bytes), std::free);
        data_ptr_ = storage_.get();
        CopyCPU(data_ptr_, data_ptr, shape_, stride_, stride_, element_size());
    }else if(device == Device::kCUDA){
#ifdef ENABLE_CUDA
        storage_ = std::shared_ptr<void>(AllocateCUDA(bytes), FreeCUDA);
        data_ptr_ = storage_.get();
        CopyHostToCUDA(data_ptr_, data_ptr, bytes);

        InitCUDAMetadata();
#else
        throw std::logic_error("Tensor constructor: CUDA support is disabled");
#endif
    }else{
        throw std::invalid_argument("Tensor constructor: unsupported device");
    }
}

Tensor::Tensor(std::vector<int> shape, Dtype dtype, Device device)
:shape_(shape), dtype_(dtype), device_(device), 
    autograd_metadata_(std::make_shared<AutogradMetadata>()){
    for (int dimension : shape_) {
        if (dimension < 0) {
            throw std::invalid_argument("Tensor dimensions must be non-negative");
        }
    }

    stride_.resize(shape_.size());
    std::size_t stride = 1;
    for (std::size_t i = shape_.size(); i > 0; --i) {
        stride_[i - 1] = stride;
        stride *= shape_[i - 1];
    }

    // shape 为 {} 表示标量，占一个元素。
    // 分配前 data_ptr_ 为空，不能直接用 numel() 判断。
    const std::size_t bytes = shape_.empty() ? element_size() : nbytes();
    if (bytes == 0) {
        throw std::invalid_argument("Tensor constructor: zero-sized shape");
    }

    if (device == Device::kCPU) {
        storage_ = std::shared_ptr<void>(AllocateCPU(bytes), std::free);
        data_ptr_ = storage_.get();
    } else if (device == Device::kCUDA) {
#ifdef ENABLE_CUDA
        storage_ = std::shared_ptr<void>(AllocateCUDA(bytes), FreeCUDA);
        data_ptr_ = storage_.get();

        InitCUDAMetadata();
#else
        throw std::logic_error("Tensor constructor: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor constructor: unsupported device");
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) { // 处理 a = a 的情况;
        return *this;
    }
    shape_ = other.shape_;
    stride_ = other.stride_;
    shape_cuda_ = other.shape_cuda_;
    stride_cuda_ = other.stride_cuda_;
    storage_ = other.storage_;
    data_ptr_ = other.data_ptr_;
    dtype_ = other.dtype_;
    device_ = other.device_;
    autograd_metadata_ = other.autograd_metadata_;
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
:   data_ptr_(other.data_ptr_),
    storage_(std::move(other.storage_)),
    shape_(std::move(other.shape_)),
    stride_(std::move(other.stride_)),
    dtype_(other.dtype_),
    device_(other.device_),
    shape_cuda_(std::move(other.shape_cuda_)),
    stride_cuda_(std::move(other.stride_cuda_)),
    autograd_metadata_(std::move(other.autograd_metadata_))
{
    other.data_ptr_ = nullptr;
    other.shape_.clear();
    other.stride_.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if(this == &other){
        return *this;
    }
    storage_ = std::move(other.storage_);
    data_ptr_ = other.data_ptr_;
    shape_ = std::move(other.shape_);
    stride_ = std::move(other.stride_);
    shape_cuda_ = std::move(other.shape_cuda_);
    stride_cuda_ = std::move(other.stride_cuda_);
    dtype_ = other.dtype_;
    device_ = other.device_;
    autograd_metadata_ = std::move(other.autograd_metadata_);

    other.data_ptr_ = nullptr;
    other.shape_.clear();
    other.stride_.clear();
    return *this;
}

bool Tensor::requires_grad() const {
    return autograd_metadata_ && autograd_metadata_->requires_grad;
}

Tensor& Tensor::requires_grad_(bool value) {
    // TODO: 实现反向传播时确定各 dtype 的梯度支持范围及检查位置。
    if (!autograd_metadata_) {
        autograd_metadata_ = std::make_shared<AutogradMetadata>();
    }
    autograd_metadata_->requires_grad = value;
    return *this;
}

bool Tensor::has_grad() const {
    return autograd_metadata_ && autograd_metadata_->grad.has_value();
}

Tensor Tensor::grad() const {
    return has_grad() ? *autograd_metadata_->grad : Tensor();
}

std::shared_ptr<AutogradNode> Tensor::grad_node() const {
    return autograd_metadata_ ? autograd_metadata_->grad_node : nullptr;
}

void Tensor::set_grad_node(std::shared_ptr<AutogradNode> node) {
    if(!autograd_metadata_){
        autograd_metadata_ = std::make_shared<AutogradMetadata>();
    }
    autograd_metadata_->grad_node = node;
}

void Tensor::accumulate_grad(const Tensor& grad) {
    // 从零开始累计梯度，如果没有的话就保存
    if (!autograd_metadata_) {
        autograd_metadata_ = std::make_shared<AutogradMetadata>();
    }

    if (!autograd_metadata_->grad.has_value()) {
        autograd_metadata_->grad = grad.clone();
    } else {
        *autograd_metadata_->grad = add(*autograd_metadata_->grad, grad);
    }
}

Tensor Tensor::detach() const {
    Tensor result;
    result.storage_ = storage_;
    result.data_ptr_ = data_ptr_;
    result.shape_ = shape_;
    result.stride_ = stride_;
    result.device_ = device_;
    result.dtype_ = dtype_;
    result.shape_cuda_ = shape_cuda_;
    result.stride_cuda_ = stride_cuda_;
    result.autograd_metadata_ = std::make_shared<AutogradMetadata>();

    return result;
}

void Tensor::backward() {
    RunBackward(*this);
}

const std::vector<int>& Tensor::shape() const {
    return shape_;
}

const std::vector<std::size_t>& Tensor::stride() const {
    return stride_;
}

int Tensor::size(std::size_t dim) const {
    if(dim >= shape_.size()){
        throw std::out_of_range("Tensor::size: dimension index out of range");
    }
    return shape_[dim];
}

int Tensor::dim() const {
    return shape_.size();
}

bool Tensor::empty() const {
    return data_ptr_ == nullptr;
}

bool Tensor::is_scalar() const {
    return !empty() && shape_.empty();
}

bool Tensor::is_contiguous() const {
    if (stride_.size() != shape_.size()) {
        return false;
    }
    std::size_t expected = 1;
    for (std::size_t i = shape_.size(); i > 0; --i) {
        if (shape_[i - 1] != 1 && stride_[i - 1] != expected) {
            return false;
        }
        expected *= static_cast<std::size_t>(shape_[i - 1]);
    }
    return true;
}

std::size_t Tensor::numel() const {
    if(shape_.empty()){
        return empty() ? 0 : 1;
    }
    return std::accumulate(
        shape_.begin(), shape_.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

std::size_t Tensor::element_size() const {
    return dtype_size(dtype_);
}

std::size_t Tensor::nbytes() const {
    return numel() * dtype_size(dtype_);
}

Dtype Tensor::dtype() const {
    return dtype_;
}

Device Tensor::device() const {
    return device_;
}

const void* Tensor::data_ptr() const {
    return data_ptr_;
}

void* Tensor::data_ptr() {
    return data_ptr_;
}

}  // namespace mtensor
