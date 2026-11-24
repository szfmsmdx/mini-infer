#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>

#include "mini_infer/core/tensor.h"
#include "tensor/check.h"
#include "mini_infer/autograd.h"
#include "ops/copy/nodes.h"
#include "ops/copy/kernels.h"
#include "device/runtime.h"
#include "tensor/tensor_impl.h"

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

    if (data_ptr() == src.data_ptr() && stride() == src.stride()) {
        return *this;
    }

    if (impl_->storage == src.impl_->storage) {
        if (!is_contiguous() || !src.is_contiguous()) {
            throw std::invalid_argument("Tensor::copy_: shared non-contiguous storage is not supported");
        }
        const auto dst_offset = impl_->storage_offset;
        const auto src_offset = src.impl_->storage_offset;
        const auto distance = dst_offset > src_offset ? dst_offset - src_offset : src_offset - dst_offset;
        if (distance < numel()) {
            throw std::invalid_argument("Tensor::copy_: overlapping storage is not supported");
        }
    }

    if (is_contiguous() && src.is_contiguous()) {
        update_version();
        CopyBytes(data_ptr(), device(), src.data_ptr(), src.device(), nbytes());
    } else if (device() == Device::kCPU) {
        update_version();
        CopyCPU(data_ptr(), src.data_ptr(), shape(), stride(), src.stride(), element_size());
    } else if (device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        update_version();
        CopyCUDA(data_ptr(), src.data_ptr(), PackLayout(*this), PackLayout(src), numel(), element_size());
#else
        throw std::logic_error("Tensor::copy_: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::copy_: unsupported device");
    }
    return *this;
}

Tensor Tensor::to(Device device) const {
    if (device == this->device()) {
        return *this;
    }
    if (device != Device::kCPU && device != Device::kCUDA) {
        throw std::invalid_argument("Tensor::to: unsupported device");
    }

    if (!is_contiguous()) {
        return clone().to(device);
    }

#ifdef ENABLE_CUDA
    if (empty()) return EmptyWithOptions(dtype(), device);

    Tensor result(shape(), dtype(), device);
    CopyBytes(result.data_ptr(), device, data_ptr(), this->device(), nbytes());
    if (EnableGrad() && CheckGradDtype(dtype())) {
        auto node = std::make_shared<ToDeviceNode>();
        node->inputs = {*this};
        node->input_device = this->device();
        result.set_grad_node(node);
    }
    return result;
#else
    throw std::logic_error("Tensor::to: CUDA support is disabled");
#endif
}

Tensor Tensor::to(Dtype dtype) const {
    if (dtype == this->dtype()) return *this;

    const bool to_float32 = dtype == Dtype::kFloat32 && dtype_size(this->dtype()) != 0;
    const bool from_float32 = this->dtype() == Dtype::kFloat32 && CheckGradDtype(dtype);
    if (!to_float32 && !from_float32) {
        throw std::invalid_argument("Tensor::to: unsupported dtype conversion");
    }

    if (empty()) return EmptyWithOptions(dtype, device());

    Tensor result(shape(), dtype, device());
    if (device() == Device::kCPU) {
        CastCPU(*this, result);
    } else if (device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        CastCUDA(*this, result);
#else
        throw std::logic_error("Tensor::to: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::to: unsupported device");
    }

    if (EnableGrad() && CheckGradDtype(this->dtype())) {
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

    Tensor result = empty() ? EmptyWithOptions(dtype(), device()) : Tensor(shape(), dtype(), device());
    if (!empty()) {
        if (is_contiguous()) {
            CopyBytes(result.data_ptr(), device(), data_ptr(), device(), nbytes());
        } else if (device() == Device::kCPU) {
            CopyCPU(result.data_ptr(), data_ptr(), shape(), result.stride(), stride(), element_size());
        } else {
#ifdef ENABLE_CUDA
            CopyCUDA(result.data_ptr(), data_ptr(), PackLayout(result), PackLayout(*this),
                     numel(), element_size());
#else
            throw std::logic_error("Tensor::clone: CUDA support is disabled");
#endif
        }
    }

    if(EnableGrad() && CheckGradDtype(dtype())) {
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
