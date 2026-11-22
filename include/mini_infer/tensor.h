#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mini_infer/dtype.h"
#include "mini_infer/scalar.h"

namespace mtensor{

// 防止循环引用提前声明;
class AutogradNode;
struct AutogradMetadata;

class Tensor {
public:
    // ---------------------------------------- init ----------------------------------------
    // 构造与所有权：拷贝共享存储，移动转移句柄。
    Tensor();
    ~Tensor() = default;
    Tensor(const Tensor&) = default;
    // 从数值创建 CPU 标量：浮点数默认为 Float32，整数默认为 Int64。
    explicit Tensor(const Scalar& value);
    // 从连续的 CPU 源数据复制，按 shape 和 dtype 解释，device 指定目标设备。
    Tensor(const void* data_ptr, std::vector<int> shape, Dtype dtype,Device device = Device::kCPU);
    // 按 shape、dtype、device 分配存储，供算子创建输出。
    Tensor(std::vector<int> shape, Dtype dtype, Device device = Device::kCPU);
    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // 基础属性：形状、布局、数据类型与设备。
    const std::vector<int>& shape() const;
    const std::vector<std::size_t>& stride() const;
    const int* shape_cuda() const { return shape_cuda_.get(); }
    const std::size_t* stride_cuda() const { return stride_cuda_.get(); }
    int size(std::size_t dim) const;
    int dim() const;
    std::size_t numel() const;
    bool empty() const;
    bool is_scalar() const;
    bool is_contiguous() const;
    std::size_t element_size() const;
    std::size_t nbytes() const;
    Dtype dtype() const;
    Device device() const;

    // 数据访问：data_ptr() 指向当前 Tensor 或视图的数据起点。
    void* data_ptr();
    const void* data_ptr() const;

public:
    // ---------------------------------------- auto_grad ----------------------------------------
    // Autograd：requires_grad 只控制是否保存梯度，不控制建图或传递。
    bool requires_grad() const;
    Tensor& requires_grad_(bool value);
    bool has_grad() const;
    Tensor grad() const;
    void zero_grad();        // 已有梯度清零并保留存储；无梯度时不操作
    Tensor detach() const;   // 共享 storage，使用新的空 meta
    void backward();         // 标量反向入口
    void backward(const Tensor& gradient);  // 指定上游梯度，须同 shape/dtype/device
    // Autograd 内部接口：供算子和反向引擎使用。
    std::shared_ptr<AutogradNode> grad_node() const;
    void set_grad_node(std::shared_ptr<AutogradNode> node);
    void accumulate_grad(const Tensor& grad);
    std::uint64_t version() const;
    void update_version();   // 原地写入前递增；直接通过 data_ptr 写入时也须调用

public:
    // ---------------------------------------- tensor_ops ----------------------------------------
    void print() const;

    // Tensor 操作：索引、视图与布局转换。
    Tensor operator[](int index) const;
    Tensor reshape(std::vector<int> shape) const;
    // 广播到目标形状，保持 dtype 和 device 不变。
    Tensor broadcast(const std::vector<int>& target_shape) const;

    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor contiguous() const;
    Tensor squeeze(int dim) const;
    Tensor unsqueeze(int dim) const;

    // 在一个维度上取连续区间，等价于 Python 的 x[..., start:start+length, ...]。
    // 返回共享 storage_ 的视图，不复制数据；修改结果会影响原 Tensor。
    Tensor narrow(int dim, int start, int length) const;

    // Tensor 操作：逐元素运算符。
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    // 标量按当前 Tensor 的 dtype 转换，规则与 fill_ 相同。
    Tensor operator+(const Scalar& value) const;
    Tensor operator-(const Scalar& value) const;
    Tensor operator*(const Scalar& value) const;
    Tensor operator/(const Scalar& value) const;
    Tensor operator-() const;
    Tensor sqrt() const;
    Tensor rsqrt() const;
    Tensor exp() const;
    Tensor log() const;

    Tensor matmul(const Tensor& other) const;

    // Tensor 操作：求和、均值与极值归约。
    Tensor sum() const;
    Tensor sum(int dim, bool keepdim = false) const;
    Tensor mean() const;
    Tensor mean(int dim, bool keepdim = false) const;
    Tensor max() const;
    Tensor max(int dim, bool keepdim = false) const;
    Tensor min() const;
    Tensor min(int dim, bool keepdim = false) const;

    // Tensor 操作：填充、复制、设备迁移与 dtype 转换。
    Tensor& fill_(const Scalar& value);
    Tensor& fill_(const Tensor& value);
    Tensor& zero_();
    Tensor& copy_(const Tensor& src);
    Tensor clone() const;
    Tensor to(Device device) const;
    Tensor to(Dtype dtype) const;

private:
    friend struct TensorMetadataTestAccess;

    // 内部辅助函数。
    Tensor& FillValue(const void* host_value);
    void InitCUDAMetadata();
    static std::vector<std::size_t> ContiguousStride(const std::vector<int>& shape);
    Tensor MakeView(std::vector<int> shape, std::vector<std::size_t> stride,
                    std::size_t byte_offset = 0) const;

    // 存储所有权、当前数据位置和 Tensor 元数据。
    void* data_ptr_ = nullptr;
    std::shared_ptr<void> storage_;
    // 共享存储的视图和 detach 共享版本，独立分配使用新版本。
    std::shared_ptr<std::uint64_t> version_ = std::make_shared<std::uint64_t>(0);
    std::vector<int> shape_;
    std::vector<std::size_t> stride_;
    Dtype dtype_ = Dtype::kFloat32;
    Device device_ = Device::kCPU;

    // gpu metadata，通过指针管理地址
    std::shared_ptr<int> shape_cuda_;
    std::shared_ptr<std::size_t> stride_cuda_;

    std::shared_ptr<AutogradMetadata> autograd_metadata_;
};

Tensor operator+(const Scalar& value, const Tensor& x);
Tensor operator-(const Scalar& value, const Tensor& x);
Tensor operator*(const Scalar& value, const Tensor& x);
Tensor operator/(const Scalar& value, const Tensor& x);

// 创建指定形状的 Tensor，并用普通数值或零维 Tensor 填充。
Tensor full(const std::vector<int>& shape, const Scalar& value, Dtype dtype,
            Device device = Device::kCPU);
Tensor full(const std::vector<int>& shape, const Tensor& value, Dtype dtype,
            Device device = Device::kCPU);

}  // namespace mtensor
