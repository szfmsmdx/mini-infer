#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/tensor.h"
#include "autograd/autograd.h"
#include "autograd/backward_nodes/tensor_ops.h"
#include "ops/elementwise.h"
#include "ops/backend.h"
#include "utils/memory.h"

namespace mtensor {

namespace {
std::vector<std::size_t> ContiguousStride(const std::vector<int>& shape) {
    std::vector<std::size_t> result(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i > 0; --i) {
        result[i - 1] = stride;
        stride *= static_cast<std::size_t>(shape[i - 1]);
    }
    return result;
}
}

Tensor Tensor::operator[](int index) const {
    if(shape_.empty() || index < 0 || index >= size(0)){
        throw std::out_of_range("Tensor::operator[]: index out of range");
    }
    if (data_ptr_ == nullptr) {
        throw std::logic_error("Tensor::operator[]: null data pointer");
    }

    // TODO: 实现 ViewNode 后移除保护，使索引视图可微分。
    if (EnableGrad() && (requires_grad() || grad_node())) {
        throw std::logic_error("Tensor::operator[]: autograd through views is not implemented");
    }

    const std::size_t offset = static_cast<std::size_t>(index) * this->stride_[0] * element_size();
    Tensor result;
    result.device_ = device_, result.dtype_ = dtype_;
    result.shape_.assign(shape_.begin() + 1, shape_.end());
    result.stride_.assign(stride_.begin() + 1, stride_.end());
    result.data_ptr_ = static_cast<std::byte*>(data_ptr_) + offset;
    result.storage_ = storage_;
    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }

    return result;
}

void Tensor::print() const {
    /*
    使用示例（a 为已构造的 CPU Tensor）：
        a.print();
    */
    const std::size_t count = numel();
    if (count > 0 && data_ptr_ == nullptr) {
        throw std::logic_error("Tensor::print: null data pointer");
    }

    if (device_ == Device::kCPU) {
        PrintCPU(data_ptr_, shape_, stride_, dtype_, empty());
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        if (!is_contiguous()) {
            clone().print();
            return;
        }
        const std::size_t bytes = nbytes();
        std::vector<std::byte> host_buffer(bytes);
        if (bytes > 0) {
            CopyCUDAToHost(host_buffer.data(), data_ptr_, bytes);
        }
        PrintCPU(host_buffer.data(), shape_, stride_, dtype_, empty());
#else
        throw std::logic_error("Tensor::print: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::print: unsupported device");
    }
}

Tensor& Tensor::fill_(const Scalar& value) {
    if (numel() == 0) {
        return *this;
    }
    if (data_ptr_ == nullptr) {
        throw std::logic_error("Tensor::fill_: null data pointer");
    }

    switch (dtype_) {
        case Dtype::kFloat32: {
            const float converted = value.to<float>();
            return FillValue(&converted);
        }
        case Dtype::kFloat16: {
            const Float16 converted(value.to<float>());
            return FillValue(&converted.bits);
        }
        case Dtype::kBFloat16: {
            const BFloat16 converted(value.to<float>());
            return FillValue(&converted.bits);
        }
        case Dtype::kInt8: {
            const std::int8_t converted = value.to<std::int8_t>();
            return FillValue(&converted);
        }
        case Dtype::kUInt8: {
            const std::uint8_t converted = value.to<std::uint8_t>();
            return FillValue(&converted);
        }
        case Dtype::kInt16: {
            const std::int16_t converted = value.to<std::int16_t>();
            return FillValue(&converted);
        }
        case Dtype::kInt32: {
            const std::int32_t converted = value.to<std::int32_t>();
            return FillValue(&converted);
        }
        case Dtype::kInt64: {
            const std::int64_t converted = value.to<std::int64_t>();
            return FillValue(&converted);
        }
        default: throw std::invalid_argument("unsupported dtype");
    }
}

Tensor& Tensor::fill_(const Tensor& value) {
    if (!value.is_scalar()) {
        throw std::invalid_argument("Tensor::fill_: value must be a non-empty scalar Tensor");
    }
    if (dtype_ != value.dtype_ || device_ != value.device_) {
        throw std::invalid_argument("Tensor::fill_: dtype and device must match");
    }
    if (empty()) {
        return *this;
    }

    if (device_ == Device::kCPU) {
        // 先保存填充值，避免 value 与目标共享存储时被覆盖。
        std::vector<std::byte> saved_value(element_size());
        std::memcpy(saved_value.data(), value.data_ptr_, element_size());
        FillCPU(data_ptr_, shape_, stride_, element_size(), saved_value.data());
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        if (storage_ == value.storage_) {
            NogradNode guard;
            Tensor saved_value = value.clone();
            FillCUDAFromDevice(data_ptr_, numel(), element_size(), saved_value.data_ptr_,
                               shape_cuda_.get(), stride_cuda_.get(), dim());
        } else {
            FillCUDAFromDevice(data_ptr_, numel(), element_size(), value.data_ptr_,
                               shape_cuda_.get(), stride_cuda_.get(), dim());
        }
#else
        throw std::logic_error("Tensor::fill_: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::fill_: unsupported device");
    }
    return *this;
}

Tensor& Tensor::FillValue(const void* host_value) {
    if (device_ == Device::kCPU) {
        FillCPU(data_ptr_, shape_, stride_, element_size(), host_value);
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        FillCUDA(data_ptr_, numel(), dtype_, host_value, shape_cuda_.get(), stride_cuda_.get(), dim());
#else
        throw std::logic_error("Tensor::fill_: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::fill_: unsupported device");
    }
    return *this;
}

Tensor& Tensor::zero_() {
    /*
    使用示例（a 为已构造的 CPU Tensor）：
        a.zero_();
    */
    const std::size_t bytes = nbytes();
    if(bytes > 0){
        if (device_ == Device::kCPU) {
            ZeroCPU(data_ptr_, shape_, stride_, element_size());
        } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
            ZeroCUDA(data_ptr_, shape_cuda_.get(), stride_cuda_.get(),
                     dim(), numel(), element_size());
#else
            throw std::logic_error("Tensor::zero_: CUDA support is disabled");
#endif
        } else {
            throw std::invalid_argument("Tensor::zero_: unsupported device");
        }
    }
    return *this;
}

Tensor& Tensor::copy_(const Tensor& src) {
    /*
    使用示例（a、b 是形状和类型相同的 CPU Tensor）:
        b.copy_(a);
    现有 copy_ 接口不支持数据重叠情况:
        原数据：     [0, 1, 2, 3]
        src：        [0, 1, 2]
        dst：           [1, 2, 3]
    */

    // 检查 device、shape、dtype 是否一致
    if(this->shape_ != src.shape_){
        throw std::invalid_argument("Tensor::copy_: shapes must match");
    }else if(this->device_ != src.device_){
        throw std::invalid_argument("Tensor::copy_: devices must match");
    }else if(this->dtype_ != src.dtype_){
        throw std::invalid_argument("Tensor::copy_: dtypes must match");
    }

    if (empty() != src.empty()) {
        throw std::invalid_argument("Tensor::copy_: empty states must match");
    }
    if (empty()) {  // empty = empty 的情况直接跳过
        return *this;
    }

    if (data_ptr_ == src.data_ptr_ && stride_ == src.stride_) {
        return *this;
    }

    if (storage_ == src.storage_) {
        if (!is_contiguous() || !src.is_contiguous()) {
            throw std::invalid_argument("Tensor::copy_: shared non-contiguous storage is not supported");
        }
        const auto* storage_ptr = static_cast<const std::byte*>(storage_.get());
        const auto dst_offset = static_cast<std::size_t>(static_cast<const std::byte*>(data_ptr_) - storage_ptr);
        const auto src_offset = static_cast<std::size_t>(static_cast<const std::byte*>(src.data_ptr_) - storage_ptr);
        const std::size_t distance = dst_offset > src_offset ? dst_offset - src_offset : src_offset - dst_offset;
        if (distance < nbytes()) {
            throw std::invalid_argument("Tensor::copy_: overlapping storage is not supported");
        }
    }

    if (device_ == Device::kCPU) {
        CopyCPU(data_ptr_, src.data_ptr_, shape_, stride_, src.stride_, element_size());
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        CopyCUDA(data_ptr_, src.data_ptr_, shape_cuda_.get(), stride_cuda_.get(),
                 src.stride_cuda_.get(), dim(), numel(), element_size());
#else
        throw std::logic_error("Tensor::copy_: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::copy_: unsupported device");
    }
    return *this;
}

Tensor Tensor::to(Device device) const {
    if (device == device_) {
        return *this;
    }
    if (device != Device::kCPU && device != Device::kCUDA) {
        throw std::invalid_argument("Tensor::to: unsupported device");
    }

    if (!is_contiguous()) {
        return clone().to(device);
    }

#ifdef ENABLE_CUDA
    Tensor result;
    result.shape_ = shape_;
    result.stride_ = ContiguousStride(shape_);
    result.dtype_ = dtype_;
    result.device_ = device;

    const std::size_t bytes = nbytes();
    if (bytes == 0) {
        return result;
    }
    if (device == Device::kCPU) {
        result.storage_ = std::shared_ptr<void>(AllocateCPU(bytes), std::free);
        result.data_ptr_ = result.storage_.get();
        CopyCUDAToHost(result.data_ptr_, data_ptr_, bytes);
    } else {
        result.storage_ = std::shared_ptr<void>(AllocateCUDA(bytes), FreeCUDA);
        result.data_ptr_ = result.storage_.get();
        CopyHostToCUDA(result.data_ptr_, data_ptr_, bytes);
        result.InitCUDAMetadata();
    }
    if (EnableGrad() && CheckGradDtype(dtype_)) {
        auto node = std::make_shared<ToDeviceNode>();
        node->inputs = {*this};
        result.set_grad_node(node);
    }
    return result;
#else
    throw std::logic_error("Tensor::to: CUDA support is disabled");
#endif
}

Tensor Tensor::clone() const {
    /*
    使用示例:
        Tensor b = a.clone();
    */

    Tensor result;
    result.shape_ = shape_;
    result.stride_ = ContiguousStride(shape_);
    result.dtype_ = dtype_;
    result.device_ = device_;

    const std::size_t bytes = nbytes();
    if(bytes > 0){
        if (device_ == Device::kCPU) {
            result.storage_ = std::shared_ptr<void>(AllocateCPU(bytes), std::free);
            result.data_ptr_ = result.storage_.get();
            CopyCPU(result.data_ptr_, data_ptr_, shape_, result.stride_, stride_, element_size());
        } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
            result.storage_ = std::shared_ptr<void>(AllocateCUDA(bytes), FreeCUDA);
            result.data_ptr_ = result.storage_.get();
            result.InitCUDAMetadata();
            CopyCUDA(result.data_ptr_, data_ptr_, shape_cuda_.get(),
                     result.stride_cuda_.get(), stride_cuda_.get(),
                     dim(), numel(), element_size());
#else
            throw std::logic_error("Tensor::clone: CUDA support is disabled");
#endif
        } else {
            throw std::invalid_argument("Tensor::clone: unsupported device");
        }
    }

    if(EnableGrad() && CheckGradDtype(dtype_)) {
        auto node = std::make_shared<CloneNode>();
        node->inputs = { *this };
        result.set_grad_node(node);
    }

    return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    /*
    使用示例:
        Tensor b = a.transpose(0, 1);
    */
    if (dim0 < 0 || dim1 < 0 ||
        static_cast<std::size_t>(dim0) >= shape_.size() ||
        static_cast<std::size_t>(dim1) >= shape_.size()) {
        throw std::out_of_range("Tensor::transpose: dimension index out of range");
    }

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

    // check dims
    if (dims.size() != shape_.size()) {
        throw std::invalid_argument("Tensor::permute: dimension count must match tensor rank");
    }
    std::vector<bool> seen(shape_.size(), false);
    for (int dim : dims) {
        if (dim < 0 || static_cast<std::size_t>(dim) >= shape_.size()) {
            throw std::out_of_range("Tensor::permute: dimension index out of range");
        }
        if (seen[dim]) {
            throw std::invalid_argument("Tensor::permute: duplicate dimension index");
        }
        seen[dim] = true;
    }
    // end check

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

Tensor Tensor::reshape(std::vector<int> shape) const {
    if (!is_contiguous()) {
        throw std::logic_error("Tensor::reshape: non-contiguous input is not supported");
    }
    std::size_t count = 1;
    for(auto dimension : shape){
        if(dimension <= 0){
            throw std::invalid_argument(
                "Tensor::reshape: dimension must be positive");
        }
        count *= dimension;
    }
    if(count != this->numel()){
        throw std::invalid_argument(
            "Tensor::reshape: shape must match");
    }

    // TODO: 实现 ViewNode 后移除保护，使 reshape 视图可微分。
    if (EnableGrad() && (requires_grad() || grad_node())) {
        throw std::logic_error("Tensor::reshape: autograd through views is not implemented");
    }

    Tensor result = *this;
    result.shape_ = std::move(shape);
    result.stride_ = ContiguousStride(result.shape_);
    result.autograd_metadata_ = std::make_shared<AutogradMetadata>();
    if (result.device_ == Device::kCUDA) {
        result.InitCUDAMetadata();
    }
    return result;
}

Tensor Tensor::narrow(int dim, int start, int length) const{
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape_.size()) {
        throw std::out_of_range("Tensor::narrow: dimension index out of range");
    }
    const int dimension = size(static_cast<std::size_t>(dim));
    if (start < 0 || start >= dimension) {
        throw std::out_of_range("Tensor::narrow: start index out of range");
    }
    if (length <= 0 || length > dimension - start) {
        throw std::invalid_argument("Tensor::narrow: length out of range");
    }
    if (data_ptr_ == nullptr) {
        throw std::logic_error("Tensor::narrow: null data pointer");
    }

    // TODO: 实现 ViewNode 后移除保护，使 narrow 视图可微分。
    if (EnableGrad() && (requires_grad() || grad_node())) {
        throw std::logic_error("Tensor::narrow: autograd through views is not implemented");
    }

    Tensor result;
    result.storage_ = storage_;
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
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return sub(*this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return mul(*this, other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return div(*this, other);
}

Tensor Tensor::operator-() const {
    return neg(*this);
}

Tensor full(const std::vector<int>& shape, const Scalar& value, Dtype dtype, Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(value);
    return result;
}

Tensor full(const std::vector<int>& shape, const Tensor& value, Dtype dtype, Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(value);
    return result;
}

}  // namespace mtensor
