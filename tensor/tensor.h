#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor/dtype.h"

namespace mtensor{

// CPU/CUDA 后端只借用数据指针，存储所有权仍由 Tensor 管理。
void CopyCPU(void* destination, const void* source, std::size_t bytes);
void ZeroCPU(void* data, std::size_t bytes);
void FillCPU(void* data, std::size_t count, Dtype dtype, const void* host_value);
void PrintCPU(const void* data, const std::vector<int>& shape, Dtype dtype, bool is_empty);

void CopyHostToCUDA(void* destination, const void* source, std::size_t bytes);
void CopyCUDA(void* destination, const void* source, std::size_t bytes);
void CopyCUDAToHost(void* destination, const void* source, std::size_t bytes);
void ZeroCUDA(void* data, std::size_t bytes);

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

    // 按 shape、dtype、device 分配存储，供算子创建输出。
    Tensor(std::vector<int> shape, Dtype dtype, Device device = Device::kCPU);

    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

public:
    const std::vector<int>& shape() const;
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
    const void* data_ptr() const;
    void print() const;

    // 形状与布局操作。TODO：当前只有接口，具体视图/拷贝语义和实现待补充。
    //
    // 根据新 shape 返回共享存储的视图，元素总数必须相同。
    //     Tensor b = a.reshape({3, 2});        // a: {2,3} -> b: {3,2}
    Tensor reshape(std::vector<int> shape) const;

    // 交换两个维度。
    //     Tensor b = a.transpose(0, 1);        // {2,3} -> {3,2}
    Tensor transpose(int dim0, int dim1) const;

    // 按 dims 重排所有维度。
    //     Tensor b = a.permute({1, 0});        // {2,3} -> {3,2}
    Tensor permute(const std::vector<int>& dims) const;

    // 返回连续布局的 Tensor；当前所有 Tensor 都连续，可共享存储。
    //     Tensor b = a.contiguous();
    Tensor contiguous() const;

    // 删除长度为 1 的维度。
    //     Tensor one_row(data, {1, 3}, Dtype::kFloat32);
    //     Tensor row = one_row.squeeze(0);      // {1,3} -> {3}
    Tensor squeeze(int dim) const;

    // 在指定位置插入长度为 1 的维度。
    //     Tensor b = a.unsqueeze(0);           // {2,3} -> {1,2,3}
    Tensor unsqueeze(int dim) const;

    // 在一个维度上取连续区间，等价于 Python 的 x[..., start:start+length, ...]。
    //     Tensor rows = a.narrow(0, 1, 1);     // 取第 1 行，{2,3} -> {1,3}
    //     Tensor cols = a.narrow(1, 1, 2);     // 每行第 1~2 列，{2,3} -> {2,2}
    // 返回共享 storage_ 的视图，不复制数据；修改结果会影响原 Tensor。
    Tensor narrow(int dim, int start, int length) const;

    // 逐元素运算，分别调用 add / sub / mul / div，返回独立存储。
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
    Tensor& FillValue(const void* host_value);

    void* data_ptr_ = nullptr;
    std::shared_ptr<void> storage_;
    std::vector<int> shape_;
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

    if (dtype_ == Dtype::kFloat16 || dtype_ == Dtype::kBFloat16) {
        // 只编码一次，随后为每个元素写入相同的 16 位浮点编码。
        const float input = static_cast<float>(value);
        const std::uint16_t half_input = dtype_ == Dtype::kFloat16
            ? EncodeFloat16(input) : EncodeBFloat16(input);
        return FillValue(&half_input);
    }

    DISPATCH_DTYPE(dtype_, Element, {
        const Element converted = static_cast<Element>(value);
        FillValue(&converted);
    });
    return *this;
}
}
