#include <cstddef>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/tensor.h"
#include "tensor/dtype.h"
#include "utils/memory.h"

namespace mtensor{

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

    // 传入数据且 shape 为 {} 时构造标量，占一个元素。
    // 此时成员数据指针尚未设置，不能靠 numel() 判断标量的大小。
    const std::size_t bytes = this->shape_.empty() ? element_size() : nbytes();
    if(bytes == 0 || data_ptr == nullptr){
        throw std::invalid_argument("Tensor constructor: null data or zero-sized shape");
    }

    if(device == Device::kCPU){
        storage_ = std::shared_ptr<void>(AllocateCPU(bytes), std::free);
        data_ptr_ = storage_.get();
        CopyCPU(data_ptr_, data_ptr, bytes);
    }else if(device == Device::kCUDA){
#ifdef ENABLE_CUDA
        storage_ = std::shared_ptr<void>(AllocateCUDA(bytes), FreeCUDA);
        data_ptr_ = storage_.get();
        CopyHostToCUDA(data_ptr_, data_ptr, bytes);
#else
        throw std::logic_error("Tensor constructor: CUDA support is disabled");
#endif
    }else{
        throw std::invalid_argument("Tensor constructor: unsupported device");
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) { // 处理 a = a 的情况;
        return *this;
    }
    shape_ = other.shape_;
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
    device_(other.device_),
    dtype_(other.dtype_)
{
    other.data_ptr_ = nullptr;
    other.shape_.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if(this == &other){
        return *this;
    }
    storage_ = std::move(other.storage_);
    data_ptr_ = other.data_ptr_;
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    device_ = other.device_;

    other.data_ptr_ = nullptr;
    other.shape_.clear();
    return *this;
}

const std::vector<int>& Tensor::shape() const {
    return shape_;
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
    const std::size_t dim_size = numel() / size(0);
    const std::size_t offset =
        static_cast<std::size_t>(index) * dim_size * element_size();
    Tensor result;
    result.device_ = device_, result.dtype_ = dtype_;
    result.shape_.assign(shape_.begin() + 1, shape_.end());
    result.data_ptr_ = static_cast<std::byte*>(data_ptr_) + offset;
    result.storage_ = storage_;

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
        PrintCPU(data_ptr_, shape_, dtype_, empty());
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        const std::size_t bytes = nbytes();
        std::vector<std::byte> host_buffer(bytes);
        if (bytes > 0) {
            CopyCUDAToHost(host_buffer.data(), data_ptr_, bytes);
        }
        PrintCPU(host_buffer.data(), shape_, dtype_, empty());
#else
        throw std::logic_error("Tensor::print: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::print: unsupported device");
    }
}

Tensor& Tensor::FillValue(const void* host_value) {
    if (device_ == Device::kCPU) {
        FillCPU(data_ptr_, numel(), dtype_, host_value);
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        LaunchFillKernel(data_ptr_, numel(), dtype_, host_value);
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
    // data_ptr_ 的 for 循环赋值此处不可用，因为 data_ptr_ 类型是 void，编译器不知道 for 步长是多少
    if(bytes > 0){
        if (device_ == Device::kCPU) {
            ZeroCPU(data_ptr_, bytes);
        } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
            ZeroCUDA(data_ptr_, bytes);
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
    使用示例（a、b 是形状和类型相同的 CPU Tensor）：
        b.copy_(a);
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

    if(data_ptr_ == src.data_ptr_){
        return *this;
    }

    if (device_ == Device::kCPU) {
        CopyCPU(data_ptr_, src.data_ptr_, nbytes());
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        CopyCUDA(data_ptr_, src.data_ptr_, nbytes());
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

#ifdef ENABLE_CUDA
    Tensor result;
    result.shape_ = shape_;
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
    result.dtype_ = dtype_;
    result.device_ = device_;

    const std::size_t bytes = nbytes();
    if(bytes > 0){
        if (device_ == Device::kCPU) {
            result.storage_ = std::shared_ptr<void>(AllocateCPU(bytes), std::free);
            result.data_ptr_ = result.storage_.get();
            CopyCPU(result.data_ptr_, data_ptr_, bytes);
        } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
            result.storage_ = std::shared_ptr<void>(AllocateCUDA(bytes), FreeCUDA);
            result.data_ptr_ = result.storage_.get();
            CopyCUDA(result.data_ptr_, data_ptr_, bytes);
#else
            throw std::logic_error("Tensor::clone: CUDA support is disabled");
#endif
        } else {
            throw std::invalid_argument("Tensor::clone: unsupported device");
        }
    }

    return result;
}
}
