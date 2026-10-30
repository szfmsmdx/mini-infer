#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor/dtype.h"

namespace mtensor{
class Tensor{
public:
    Tensor() = default;
    ~Tensor() = default;
    Tensor(const Tensor&) = default;

    // 从连续的 CPU 源数据复制，按 shape 和 dtype 解释，device 指定目标设备。
    Tensor(const void* data_ptr, std::vector<int> shape, Dtype dtype,Device device = Device::kCPU);

    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

public:
    const std::vector<int>& size() const;
    int size(std::size_t dim) const;
    int dim() const;
    std::size_t numel() const;
    bool empty() const;
    std::size_t element_size() const;
    std::size_t nbytes() const;
    Dtype dtype() const;
    Device device() const;

    Tensor operator[](int index) const;
    void* data_ptr();
    void print() const;

    template <typename T>
    Tensor& fill_(T value);
    Tensor& zero_();
    Tensor& copy_(const Tensor& src);
    Tensor clone() const;

private:
    void* data_ptr_ = nullptr;
    std::shared_ptr<void> storage_;
    std::vector<int> shape;
    Dtype dtype_ = Dtype::kFloat32;
    Device device_ = Device::kCPU;
};

template <typename T>
Tensor& Tensor::fill_(T value) {
    const std::size_t count = numel();
    if (count == 0) {
        return *this;
    }
    if (data_ptr_ == nullptr) {
        throw std::logic_error("Tensor::fill_: null data pointer");
    }

    if (device_ == Device::kCPU) {
        if (dtype_ == Dtype::kFloat16 || dtype_ == Dtype::kBFloat16) {
            // 只编码一次，随后为每个元素写入相同的 16 位浮点编码。
            const float input = static_cast<float>(value);
            const std::uint16_t encoded = dtype_ == Dtype::kFloat16
                ? EncodeFloat16(input) : EncodeBFloat16(input);
            std::uint16_t* ptr = static_cast<std::uint16_t*>(data_ptr_);
            for (std::size_t i = 0; i < count; ++i) {
                ptr[i] = encoded;
            }
            return *this;
        }

        MTENSOR_DISPATCH_DTYPE(dtype_, Element, {
            Element* ptr = static_cast<Element*>(data_ptr_);
            const Element converted = static_cast<Element>(value);
            for (std::size_t i = 0; i < count; ++i) {
                ptr[i] = converted;
            }
        });
    } else if (device_ == Device::kCUDA) {
        // TODO: 实现 CUDA 填充 kernel，按 dtype 转换数值并写入设备内存。
        // TODO: FP16/BF16 写入对应的 16 位编码，不能直接复用 CPU 指针解引用。
        throw std::logic_error("Tensor::fill_: CUDA is not implemented");
    } else {
        throw std::invalid_argument("Tensor::fill_: unsupported device");
    }
    return *this;
}
}
