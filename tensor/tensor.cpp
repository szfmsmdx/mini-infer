#include <cstddef>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/tensor.h"
#include "tensor/dtype.h"
#include "ops/elementwise.h"
#include "utils/memory.h"

namespace mtensor{

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

Tensor::Tensor(const void* data_ptr, std::vector<int> shape, Dtype dtype, Device device)
: shape_(std::move(shape)), dtype_(dtype), device_(device) {
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
:shape_(shape), dtype_(dtype), device_(device) {
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
    stride_cuda_(std::move(other.stride_cuda_))
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

    other.data_ptr_ = nullptr;
    other.shape_.clear();
    other.stride_.clear();
    return *this;
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

Tensor Tensor::operator[](int index) const {
    if(shape_.empty() || index < 0 || index >= size(0)){
        throw std::out_of_range("Tensor::operator[]: index out of range");
    }
    if (data_ptr_ == nullptr) {
        throw std::logic_error("Tensor::operator[]: null data pointer");
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

const void* Tensor::data_ptr() const {
    return data_ptr_;
}

void* Tensor::data_ptr() {
    return data_ptr_;
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
    Tensor result = *this;
    result.shape_ = std::move(shape);
    result.stride_ = ContiguousStride(result.shape_);
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
}
