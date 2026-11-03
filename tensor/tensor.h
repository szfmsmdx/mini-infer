#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor/dtype.h"

namespace mtensor{

void LaunchFillKernel(
    void* device_ptr,
    std::size_t count,
    Dtype dtype,
    const void* host_value);

class Tensor {
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
    /*
    使用示例（a 为已构造的 CPU Tensor）：
        a.fill_(1.0f);
    */
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
            const std::uint16_t half_input = dtype_ == Dtype::kFloat16
                ? EncodeFloat16(input) : EncodeBFloat16(input);
            std::uint16_t* ptr = static_cast<std::uint16_t*>(data_ptr_);
            for (std::size_t i = 0; i < count; ++i) {
                ptr[i] = half_input;
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
#ifdef MINI_INFER_ENABLE_CUDA
        if(dtype_ == Dtype::kBFloat16 || dtype_ == Dtype::kFloat16){
            const float input = static_cast<float>(value);
            const std::uint16_t half_input = dtype_ == Dtype::kBFloat16
                ? EncodeBFloat16(input)
                : EncodeFloat16(input);
            LaunchFillKernel(data_ptr_, count, dtype_, &half_input);
        } else {
            MTENSOR_DISPATCH_DTYPE(dtype_, Element, {
                const Element converted = static_cast<Element>(value);
                LaunchFillKernel(data_ptr_, count, dtype_, &converted);
            });
        }
#else
        throw std::logic_error("Tensor::fill_: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::fill_: unsupported device");
    }
    return *this;
}
}
