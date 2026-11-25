#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "mini_infer/tensor/tensor.h"
#include "runtime/memory.h"
#include "tensor/tensor_impl.h"

namespace mtensor {

namespace {

ShapeView CheckSource(const void* data, ShapeView shape) {
    if (!data) throw std::invalid_argument("Tensor constructor: null source data");
    return shape;
}

std::size_t CheckedMultiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("Tensor layout: size overflow");
    }
    return a * b;
}

std::size_t ElementCount(ShapeView shape) {
    std::size_t count = 1;
    for (int extent : shape) {
        if (extent <= 0) throw std::invalid_argument("Tensor layout: dimensions must be positive");
        count = CheckedMultiply(count, static_cast<std::size_t>(extent));
    }
    return count;
}

}  // namespace

void CheckLayoutRank(std::size_t rank) {
    if (rank > kMaxDim) throw std::invalid_argument("Tensor layout: rank exceeds kMaxDim");
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, ShapeView shape, StrideView stride,
                       std::size_t storage_offset, Dtype dtype, Device device,
                       std::shared_ptr<std::uint64_t> version)
    : storage(std::move(storage)), storage_offset(storage_offset), dtype(dtype),
      empty_device(device), rank(0), numel(0),
      version(version ? std::move(version) : std::make_shared<std::uint64_t>(0)) {
    CheckLayoutRank(shape.size());
    if (shape.size() != stride.size()) {
        throw std::invalid_argument("Tensor layout: shape and stride rank must match");
    }
    if (!this->storage) {
        if (!shape.empty() || storage_offset != 0) {
            throw std::invalid_argument("Tensor layout: nonempty layout requires storage");
        }
        return;
    }

    numel = ElementCount(shape);
    const auto itemsize = dtype_size(dtype);
    if (!itemsize) throw std::invalid_argument("Tensor layout: unsupported dtype");
    CheckedMultiply(numel, itemsize);
    const auto capacity = this->storage->nbytes() / itemsize;
    if (storage_offset >= capacity) {
        throw std::out_of_range("Tensor layout: storage offset out of bounds");
    }
    std::size_t remaining = capacity - 1 - storage_offset;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        const auto extent = static_cast<std::size_t>(shape[i] - 1);
        if (extent && stride[i] > remaining / extent) {
            throw std::out_of_range("Tensor layout: view exceeds storage capacity");
        }
        remaining -= extent * stride[i];
    }
    std::copy(shape.begin(), shape.end(), this->shape);
    std::copy(stride.begin(), stride.end(), this->stride);
    rank = static_cast<int>(shape.size());
}

Tensor::Tensor()
    : impl_(std::make_shared<TensorImpl>(nullptr, ShapeView{}, StrideView{}, 0,
                                         Dtype::kFloat32, Device::kCPU)) {}

Tensor Tensor::EmptyWithOptions(Dtype dtype, Device device) {
    Tensor result;
    result.impl_->dtype = dtype;
    result.impl_->empty_device = device;
    return result;
}

Tensor::Tensor(const Scalar& value)
    : Tensor(/*shape=*/ShapeView{},
             /*dtype=*/value.is_floating_point() ? Dtype::kFloat32 : Dtype::kInt64,
             /*device=*/Device::kCPU) {
    constexpr std::uint64_t max_int64 = std::numeric_limits<std::int64_t>::max();
    if (value.is_unsigned_integer() && value.to<std::uint64_t>() > max_int64) {
        throw std::out_of_range("Tensor: integer value is out of Int64 range");
    }
    fill_(value);
}

Tensor::Tensor(const void* data_ptr, ShapeView shape, Dtype dtype, Device device)
    : Tensor(CheckSource(data_ptr, shape), dtype, device) {
    CopyBytes(this->data_ptr(), device, data_ptr, Device::kCPU, nbytes());
}

Tensor::Tensor(ShapeView shape, Dtype dtype, Device device) {
    CheckLayoutRank(shape.size());
    const auto count = ElementCount(shape);
    const auto itemsize = dtype_size(dtype);
    if (!itemsize) throw std::invalid_argument("Tensor constructor: unsupported dtype");
    const auto bytes = CheckedMultiply(count, itemsize);
    std::size_t stride[kMaxDim]{};
    std::size_t step = 1;
    for (std::size_t i = shape.size(); i > 0; --i) {
        stride[i - 1] = step;
        step *= static_cast<std::size_t>(shape[i - 1]);
    }
    auto storage = std::make_shared<Storage>(GetAllocator(device).allocate(bytes), bytes);
    impl_ = std::make_shared<TensorImpl>(std::move(storage), shape,
                                        StrideView(stride, shape.size()), 0, dtype, device);
}

Tensor::Tensor(Tensor&& other) noexcept {
    *this = std::move(other);
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        moved_from_dtype_ = other.moved_from_dtype_ = other.dtype();
        moved_from_device_ = other.moved_from_device_ = other.device();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

std::uint64_t Tensor::version() const { return impl_ ? *impl_->version : 0; }
void Tensor::update_version() { if (impl_) ++(*impl_->version); }

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

ShapeView Tensor::shape() const { return impl_ ? ShapeView(impl_->shape, impl_->rank) : ShapeView{}; }
StrideView Tensor::stride() const { return impl_ ? StrideView(impl_->stride, impl_->rank) : StrideView{}; }

int Tensor::size(std::size_t dim) const {
    if (dim >= shape().size()) throw std::out_of_range("Tensor::size: dimension index out of range");
    return shape()[dim];
}

int Tensor::dim() const { return impl_ ? impl_->rank : 0; }
bool Tensor::empty() const { return !impl_ || !impl_->storage; }
bool Tensor::is_scalar() const { return !empty() && dim() == 0; }

bool Tensor::is_contiguous() const {
    std::size_t expected = 1;
    for (int i = dim() - 1; i >= 0; --i) {
        if (impl_->shape[i] != 1 && impl_->stride[i] != expected) return false;
        expected *= static_cast<std::size_t>(impl_->shape[i]);
    }
    return true;
}

std::size_t Tensor::numel() const { return impl_ ? impl_->numel : 0; }
std::size_t Tensor::element_size() const { return dtype_size(dtype()); }
std::size_t Tensor::nbytes() const { return numel() * element_size(); }
Dtype Tensor::dtype() const { return impl_ ? impl_->dtype : moved_from_dtype_; }
Device Tensor::device() const {
    if (!impl_) return moved_from_device_;
    return impl_->storage ? impl_->storage->device() : impl_->empty_device;
}

bool Tensor::shares_storage(const Tensor& other) const {
    return impl_ && other.impl_ && impl_->storage &&
           impl_->storage == other.impl_->storage;
}

const void* Tensor::data_ptr() const {
    if (empty()) return nullptr;
    return static_cast<const std::byte*>(impl_->storage->data()) +
           impl_->storage_offset * element_size();
}

void* Tensor::data_ptr() { return const_cast<void*>(static_cast<const Tensor&>(*this).data_ptr()); }

}  // namespace mtensor
