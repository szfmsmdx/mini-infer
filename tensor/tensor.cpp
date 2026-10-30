#include "tensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensor/dtype.h"
#include "utils/logger.h"
#include "utils/memory.h"

namespace mtensor{

namespace {

float DecodeFloat16(const void* element) {
    std::uint16_t bits;
    std::memcpy(&bits, element, sizeof(bits));

    const bool negative = (bits & 0x8000u) != 0;
    const int exponent = (bits >> 10) & 0x1f;
    const int fraction = bits & 0x03ff;

    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(fraction), -24);
    } else if (exponent == 0x1f) {
        value = fraction == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else {
        // 普通 Float16: (1 + fraction / 1024) * 2^(exponent - 15)。
        // 改写为 (1024 + fraction) * 2^(exponent - 25)，其中 25 = 10 + 15。
        value = std::ldexp(static_cast<float>(0x400 + fraction), exponent - 25);
    }
    return negative ? -value : value;
}

float DecodeBFloat16(const void* element) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::uint16_t bits;
    std::memcpy(&bits, element, sizeof(bits));
    const std::uint32_t float_bits = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &float_bits, sizeof(value));    // 只取前 16bit
    return value;
}

void PrintElement(std::ostream& out, Dtype dtype, const void* element) {
    switch (dtype) {
        case Dtype::kFloat32:
            out << *static_cast<const float*>(element);
            return;
        case Dtype::kFloat16:
            out << DecodeFloat16(element);
            return;
        case Dtype::kBFloat16:
            out << DecodeBFloat16(element);
            return;
        case Dtype::kInt8:
            out << static_cast<int>(*static_cast<const std::int8_t*>(element));
            return;
        case Dtype::kUInt8:
            out << static_cast<unsigned int>(*static_cast<const std::uint8_t*>(element));
            return;
        case Dtype::kInt16:
            out << *static_cast<const std::int16_t*>(element);
            return;
        case Dtype::kInt32:
            out << *static_cast<const std::int32_t*>(element);
            return;
        case Dtype::kInt64:
            out << *static_cast<const std::int64_t*>(element);
            return;
    }
    throw std::invalid_argument("Tensor::print: unsupported dtype");
}

void PrintNested(std::ostream& out, Dtype dtype, const std::byte* bytes,
                 const std::vector<int>& shape, std::size_t dimension,
                 std::size_t element_bytes, std::size_t& flat_index) {
    out << '[';
    for (int index = 0; index < shape[dimension]; ++index) {
        if (index > 0) {
            out << ',';
            if (dimension + 1 == shape.size()) {
                out << ' ';
            } else {
                // Align the next slice under the opening '[' after "tensor(".
                out << '\n' << std::string(8 + dimension, ' ');
            }
        }

        if (dimension + 1 == shape.size()) {
            PrintElement(out, dtype, bytes + flat_index * element_bytes);
            ++flat_index;
        } else {
            PrintNested(out, dtype, bytes, shape, dimension + 1,
                        element_bytes, flat_index);
        }
    }
    out << ']';
}

}  // namespace

Tensor::Tensor(const void* data_ptr, std::vector<int> shape, Dtype dtype, Device device)
: shape(std::move(shape)), dtype_(dtype), device_(device) {
    for (int dimension : this->shape) {
        if (dimension < 0) {
            Logger::error("Tensor shape contains a negative dimension");
            throw std::invalid_argument("Tensor dimensions must be non-negative");
        }
    }

    // 传入数据且 shape 为 {} 时构造标量，占一个元素。
    // 此时成员数据指针尚未设置，不能靠 numel() 判断标量的大小。
    const std::size_t bytes = this->shape.empty() ? element_size() : nbytes();
    if(bytes == 0 || data_ptr == nullptr){
        Logger::error("Init Tensor With Bad Input");
        throw std::invalid_argument("null score data!");
    }

    if(device == Device::kCPU){
        storage_ = std::shared_ptr<void>(AllocateCPU(bytes), std::free);
        data_ptr_ = storage_.get();
        std::memcpy(data_ptr_, data_ptr, bytes);
    }else if(device == Device::kCUDA){
        // TODO: 使用 cudaMalloc 分配显存，storage_ 绑定 cudaFree 删除器。
        // TODO: 设置 data_ptr_，将 CPU 源数据通过 cudaMemcpyHostToDevice 复制到显存。
        throw std::logic_error("Tensor constructor: CUDA is not implemented");
    }else{
        throw std::invalid_argument("Tensor constructor: unsupported device");
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) { // 处理 a = a 的情况;
        return *this;
    }
    shape = other.shape;
    storage_ = other.storage_;
    data_ptr_ = other.data_ptr_;
    dtype_ = other.dtype_;
    device_ = other.device_;
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
:   data_ptr_(other.data_ptr_),
    storage_(std::move(other.storage_)),
    shape(std::move(other.shape)),
    device_(other.device_),
    dtype_(other.dtype_)
{
    other.data_ptr_ = nullptr;
    other.shape.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if(this == &other){
        return *this;
    }
    storage_ = std::move(other.storage_);
    data_ptr_ = other.data_ptr_;
    shape = std::move(other.shape);
    dtype_ = other.dtype_;
    device_ = other.device_;

    other.data_ptr_ = nullptr;
    other.shape.clear();
    return *this;
}

const std::vector<int>& Tensor::size() const {
    return shape;
}

int Tensor::size(std::size_t dim) const {
    if(dim >= shape.size()){
        Logger::error("Bad dim to get size !!!");
        throw std::out_of_range("dim index out of range!!!");
    }
    return shape[dim];
}

int Tensor::dim() const {
    return shape.size();
}

bool Tensor::empty() const {
    return data_ptr_ == nullptr;
}

std::size_t Tensor::numel() const {
    if(shape.empty()){
        return empty() ? 0 : 1;
    }
    return std::accumulate(
        shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
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
    if(shape.empty() || index < 0 || index >= size(0)){
        Logger::error("Bad dim to get index");
        throw std::out_of_range("Tensor index out of range!!!");
    }
    if (data_ptr_ == nullptr) {
        throw std::logic_error("Tensor::operator[]: null data pointer");
    }
    const std::size_t dim_size = numel() / size(0);
    const std::size_t offset =
        static_cast<std::size_t>(index) * dim_size * element_size();
    Tensor result;
    result.device_ = device_, result.dtype_ = dtype_;
    result.shape.assign(shape.begin() + 1, shape.end());
    result.data_ptr_ = static_cast<std::byte*>(data_ptr_) + offset;
    result.storage_ = storage_;

    return result;
}

void* Tensor::data_ptr() {
    return data_ptr_;
}

void Tensor::print() const {
    const std::size_t count = numel();
    if (count > 0 && data_ptr_ == nullptr) {
        throw std::logic_error("Tensor::print: null data pointer");
    }

    if (device_ == Device::kCPU) {
        std::cout << "tensor(";
        if (shape.empty()) {
            if (empty()) {
                std::cout << "[]";
            } else {    // 标量 shape empty 但是 tensor 不 empty
                PrintElement(std::cout, dtype_, data_ptr_);
            }
        } else {
            const auto* bytes = static_cast<const std::byte*>(data_ptr_);
            std::size_t flat_index = 0;
            PrintNested(std::cout, dtype_, bytes, shape, 0, element_size(), flat_index);
        }
        std::cout << ")\n";
    } else if (device_ == Device::kCUDA) {
        // TODO: 用 cudaMemcpyDeviceToHost 复制到 CPU Tensor，再复用 CPU 打印。
        throw std::logic_error("Tensor::print: CUDA is not implemented");
    } else {
        throw std::invalid_argument("Tensor::print: unsupported device");
    }
}

Tensor& Tensor::zero_() {
    const std::size_t bytes = nbytes();
    // data_ptr_ 的 for 循环赋值此处不可用，因为 data_ptr_ 类型是 void，编译器不知道 for 步长是多少
    if(bytes > 0){
        if (device_ == Device::kCPU) {
            std::memset(data_ptr_, 0, bytes);
        } else if (device_ == Device::kCUDA) {
            // TODO: 使用 cudaMemset(data_ptr_, 0, bytes) 清零设备内存。
            throw std::logic_error("Tensor::zero_: CUDA is not implemented");
        } else {
            throw std::invalid_argument("Tensor::zero_: unsupported device");
        }
    }
    return *this;
}

Tensor& Tensor::copy_(const Tensor& src) {
    if(this->shape != src.shape){
        Logger::error("Copy should get same shape!!!");
        throw std::invalid_argument("Tensor::copy_: shapes must match");
    }else if(this->device_ != src.device_){
        Logger::error("Copy should get same device!!!");
        throw std::invalid_argument("Tensor::copy_: devices must match");
    }else if(this->dtype_ != src.dtype_){
        Logger::error("Copy should get same dtype!!!");
        throw std::invalid_argument("Tensor::copy_: dtypes must match");
    }

    if (empty() != src.empty()) {
        Logger::error("Cannot copy between an empty Tensor and a non-empty Tensor");
        throw std::invalid_argument("Tensor::copy_: empty states must match");
    }
    if (empty()) {  // empty = empty 的情况直接跳过
        return *this;
    }

    if(data_ptr_ == src.data_ptr_){
        return *this;
    }

    if (device_ == Device::kCPU) {
        std::memcpy(data_ptr_, src.data_ptr_, nbytes());
    } else if (device_ == Device::kCUDA) {
        // TODO: 使用 cudaMemcpyDeviceToDevice 复制；当前接口要求源、目标 device 相同。
        throw std::logic_error("Tensor::copy_: CUDA is not implemented");
    } else {
        throw std::invalid_argument("Tensor::copy_: unsupported device");
    }
    return *this;
}

Tensor Tensor::clone() const {
    Tensor result;
    result.shape = shape;
    result.dtype_ = dtype_;
    result.device_ = device_;

    const std::size_t bytes = nbytes();
    if(bytes > 0){
        if (device_ == Device::kCPU) {
            result.storage_ = std::shared_ptr<void>(AllocateCPU(bytes), std::free);
            result.data_ptr_ = result.storage_.get();
            std::memcpy(result.data_ptr_, data_ptr_, bytes);
        } else if (device_ == Device::kCUDA) {
            // TODO: 分配独立显存并绑定 cudaFree 删除器，再执行 cudaMemcpyDeviceToDevice。
            // TODO: 复制源使用当前 data_ptr_，保留对索引视图的 clone 支持。
            throw std::logic_error("Tensor::clone: CUDA is not implemented");
        } else {
            throw std::invalid_argument("Tensor::clone: unsupported device");
        }
    }

    return result;
}
}
