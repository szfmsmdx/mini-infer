#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor/dtype.h"

namespace mtensor{

// CPU/CUDA 后端只借用数据指针，存储所有权仍由 Tensor 管理。
void CopyCPU(void* dst_data_ptr, const void* src_data_ptr, const std::vector<int>& shape,
             const std::vector<std::size_t>& dst_stride,
             const std::vector<std::size_t>& src_stride, std::size_t element_size);
void ZeroCPU(void* data_ptr, const std::vector<int>& shape,
             const std::vector<std::size_t>& stride, std::size_t element_size);
void FillCPU(void* data_ptr, const std::vector<int>& shape,
             const std::vector<std::size_t>& stride, std::size_t element_size,
             const void* host_value);
void PrintCPU(const void* data_ptr, const std::vector<int>& shape,
              const std::vector<std::size_t>& stride, Dtype dtype, bool is_empty);

void CopyHostToCUDA(void* destination, const void* source, std::size_t bytes);
void CopyCUDA(void* destination, const void* source, const int* shape,
                     const std::size_t* dst_stride, const std::size_t* src_stride,
                     int ndim, std::size_t numel, std::size_t element_size);
void CopyCUDAToHost(void* destination, const void* source, std::size_t bytes);
void ZeroCUDA(void* data_ptr, const int* shape_cuda, const std::size_t* stride_cuda,
              int ndim, std::size_t numel, std::size_t element_size);

void FillCUDA(
    void* device_ptr,
    std::size_t numel,
    Dtype dtype,
    const void* host_value,
    const int* shape,
    const std::size_t* stride,
    int ndim
);

// 防止循环引用提前声明;
class AutogradNode;
struct AutogradMetadata;

class Tensor {
public:
    Tensor();
    ~Tensor() = default;
    Tensor(const Tensor&) = default;

    // 从连续的 CPU 源数据复制，按 shape 和 dtype 解释，device 指定目标设备。
    Tensor(const void* data_ptr, std::vector<int> shape, Dtype dtype,Device device = Device::kCPU);

    // 按 shape、dtype、device 分配存储，供算子创建输出。
    Tensor(std::vector<int> shape, Dtype dtype, Device device = Device::kCPU);

    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

public:
    // 基础 tensor 方法
    const std::vector<int>& shape() const;
    const std::vector<std::size_t>& stride() const;
    int size(std::size_t dim) const;
    int dim() const;
    std::size_t numel() const;
    bool empty() const;
    bool is_contiguous() const;
    std::size_t element_size() const;
    std::size_t nbytes() const;
    Dtype dtype() const;
    Device device() const;
    bool requires_grad() const;
    Tensor& requires_grad_(bool value);
    bool has_grad() const;
    Tensor grad() const;

    // Autograd 内部接口；requires_grad 只控制是否保存梯度，不控制建图。
    std::shared_ptr<AutogradNode> grad_node() const;
    void set_grad_node(std::shared_ptr<AutogradNode> node);
    void accumulate_grad(const Tensor& grad);
    Tensor detach() const;   // 共享 storage，使用新的空 meta
    void backward();         // 标量反向入口

    const int* shape_cuda() const { return shape_cuda_.get(); }
    const std::size_t* stride_cuda() const { return stride_cuda_.get(); }

    Tensor operator[](int index) const;
    void* data_ptr();
    const void* data_ptr() const;
    void print() const;

    Tensor reshape(std::vector<int> shape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor contiguous() const;
    Tensor squeeze(int dim) const;
    Tensor unsqueeze(int dim) const;

    // 在一个维度上取连续区间，等价于 Python 的 x[..., start:start+length, ...]。
    // 返回共享 storage_ 的视图，不复制数据；修改结果会影响原 Tensor。
    Tensor narrow(int dim, int start, int length) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    template <typename T>
    Tensor& fill_(T value);
    Tensor& zero_();
    Tensor& copy_(const Tensor& src);
    Tensor clone() const;
    Tensor to(Device device) const;

private:
    friend struct TensorMetadataTestAccess;
    Tensor& FillValue(const void* host_value);
    void InitCUDAMetadata();

    void* data_ptr_ = nullptr;
    std::shared_ptr<void> storage_;
    std::vector<int> shape_;
    std::vector<std::size_t> stride_;
    Dtype dtype_ = Dtype::kFloat32;
    Device device_ = Device::kCPU;

    // gpu metadata，通过指针管理地址
    std::shared_ptr<int> shape_cuda_;
    std::shared_ptr<std::size_t> stride_cuda_;

    std::shared_ptr<AutogradMetadata> autograd_metadata_;
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

    switch (dtype_) {
        case Dtype::kFloat32: {
            const float converted = static_cast<float>(value);
            return FillValue(&converted);
        }
        case Dtype::kFloat16: {
            const Float16 converted(static_cast<float>(value));
            return FillValue(&converted.bits);
        }
        case Dtype::kBFloat16: {
            const BFloat16 converted(static_cast<float>(value));
            return FillValue(&converted.bits);
        }
        case Dtype::kInt8: {
            const std::int8_t converted = static_cast<std::int8_t>(value);
            return FillValue(&converted);
        }
        case Dtype::kUInt8: {
            const std::uint8_t converted = static_cast<std::uint8_t>(value);
            return FillValue(&converted);
        }
        case Dtype::kInt16: {
            const std::int16_t converted = static_cast<std::int16_t>(value);
            return FillValue(&converted);
        }
        case Dtype::kInt32: {
            const std::int32_t converted = static_cast<std::int32_t>(value);
            return FillValue(&converted);
        }
        case Dtype::kInt64: {
            const std::int64_t converted = static_cast<std::int64_t>(value);
            return FillValue(&converted);
        }
        default: throw std::invalid_argument("unsupported dtype");
    }
}
}
