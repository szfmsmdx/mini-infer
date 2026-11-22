#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>

#include "mini_infer/tensor.h"
#include "tensor/check.h"
#include "mini_infer/autograd.h"
#include "ops/copy/nodes.h"
#include "ops/copy/kernels.h"
#include "device/memory.h"

namespace mtensor {

Tensor& Tensor::copy_(const Tensor& src) {
    /*
    使用示例（a、b 是形状和类型相同的 CPU Tensor）:
        b.copy_(a);
    现有 copy_ 接口不支持数据重叠情况:
        原数据：     [0, 1, 2, 3]
        src：        [0, 1, 2]
        dst：           [1, 2, 3]
    */

    CheckSameShape(*this, src, "Tensor::copy_: shapes must match");
    CheckSameDevice(*this, src, "Tensor::copy_: devices must match");
    CheckSameDtype(*this, src, "Tensor::copy_: dtypes must match");

    if (empty() != src.empty()) {
        throw std::invalid_argument("Tensor::copy_: empty states must match");
    }
    if (empty()) {  // empty = empty 的情况直接跳过
        return *this;
    }
    CheckBroadcastWritable(*this);

    if (data_ptr_ == src.data_ptr_ && stride_ == src.stride_) {
        return *this;
    }

    if (storage_ == src.storage_) {
        if (!is_contiguous() || !src.is_contiguous()) {
            throw std::invalid_argument("Tensor::copy_: shared non-contiguous storage is not supported");
        }
        const auto* storage_ptr = static_cast<const std::byte*>(storage_.get());
        const auto* dst_bytes = static_cast<const std::byte*>(data_ptr_);
        const auto* src_bytes = static_cast<const std::byte*>(src.data_ptr_);
        const auto dst_offset = static_cast<std::size_t>(dst_bytes - storage_ptr);
        const auto src_offset = static_cast<std::size_t>(src_bytes - storage_ptr);
        const std::size_t distance = dst_offset > src_offset ? dst_offset - src_offset : src_offset - dst_offset;
        if (distance < nbytes()) {
            throw std::invalid_argument("Tensor::copy_: overlapping storage is not supported");
        }
    }

    if (device_ == Device::kCPU) {
        update_version();
        CopyCPU(data_ptr_, src.data_ptr_, shape_, stride_, src.stride_, element_size());
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        update_version();
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
        result.storage_ = AllocateStorage(bytes, Device::kCPU);
        result.data_ptr_ = result.storage_.get();
        CopyCUDAToHost(result.data_ptr_, data_ptr_, bytes);
    } else {
        result.storage_ = AllocateStorage(bytes, Device::kCUDA);
        result.data_ptr_ = result.storage_.get();
        CopyHostToCUDA(result.data_ptr_, data_ptr_, bytes);
        result.InitCUDAMetadata();
    }
    if (EnableGrad() && CheckGradDtype(dtype_)) {
        auto node = std::make_shared<ToDeviceNode>();
        node->inputs = {*this};
        node->input_device = device_;
        result.set_grad_node(node);
    }
    return result;
#else
    throw std::logic_error("Tensor::to: CUDA support is disabled");
#endif
}

Tensor Tensor::to(Dtype dtype) const {
    if (dtype == dtype_) return *this;

    const bool to_float32 = dtype == Dtype::kFloat32 && dtype_size(dtype_) != 0;
    const bool from_float32 = dtype_ == Dtype::kFloat32 && CheckGradDtype(dtype);
    if (!to_float32 && !from_float32) {
        throw std::invalid_argument("Tensor::to: unsupported dtype conversion");
    }

    if (empty()) {
        Tensor result;
        result.dtype_ = dtype;
        result.device_ = device_;
        return result;
    }

    Tensor result(shape_, dtype, device_);
    if (device_ == Device::kCPU) {
        CastCPU(*this, result);
    } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
        CastCUDA(*this, result);
#else
        throw std::logic_error("Tensor::to: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::to: unsupported device");
    }

    if (EnableGrad() && CheckGradDtype(dtype_)) {
        auto node = std::make_shared<ToDtypeNode>();
        node->inputs = {*this};
        result.set_grad_node(node);
    }
    return result;
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
            result.storage_ = AllocateStorage(bytes, Device::kCPU);
            result.data_ptr_ = result.storage_.get();
            CopyCPU(result.data_ptr_, data_ptr_, shape_, result.stride_, stride_, element_size());
        } else if (device_ == Device::kCUDA) {
#ifdef ENABLE_CUDA
            result.storage_ = AllocateStorage(bytes, Device::kCUDA);
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


std::vector<Tensor> CloneNode::backward(const Tensor& output_grad) {
    return { output_grad };
}

std::vector<Tensor> ToDeviceNode::backward(const Tensor& output_grad) {
    // auto_grad 引擎确保不会循环建图
    return {output_grad.to(input_device)};
}

std::vector<Tensor> ToDtypeNode::backward(const Tensor& output_grad) {
    return {output_grad.to(inputs[0].dtype())};
}

}  // namespace mtensor
