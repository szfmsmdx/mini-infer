#include <memory>
#include <stdexcept>
#include <vector>

#include "mini_infer/ops/reduction.h"
#include "tensor/check.h"
#include "mini_infer/autograd.h"
#include "ops/reduction/nodes.h"
#include "ops/reduction/kernels.h"
#include "mini_infer/ops/elementwise.h"

namespace mtensor {

Tensor sum(const Tensor& input) {
    CheckNotEmpty(input, "sum: input must not be empty");
    const bool floating = CheckGradDtype(input.dtype());
    Tensor result(/*shape=*/std::vector<int>{},
                  /*dtype=*/floating ? input.dtype() : Dtype::kInt64,
                  /*device=*/input.device());
    if (input.device() == Device::kCPU) {
        SumCPU(input.data_ptr(), result.data_ptr(), input.dtype(), input.shape(),
               input.stride(), input.numel());
    } else if (input.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        SumCUDA(input.data_ptr(), result.data_ptr(), input.dtype(), input.shape_cuda(),
                input.stride_cuda(), input.dim(), input.numel());
#else
        throw std::logic_error("sum: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("sum: unsupported device");
    }

    if (EnableGrad() && floating) {
        auto node = std::make_shared<SumNode>();
        node->inputs = {input};
        result.set_grad_node(node);
    }
    return result;
}

Tensor sum(const Tensor& input, int dim, bool keepdim) {
    CheckNotEmpty(input, "sum: input must not be empty");
    if (dim < 0) dim += input.dim();
    CheckDim(dim, input.shape().size(), "sum: dim out of range");

    const bool floating = CheckGradDtype(input.dtype());

    auto output_shape = input.shape();
    if (keepdim) output_shape[dim] = 1;
    else output_shape.erase(output_shape.begin() + dim);
    
    Tensor result(output_shape, floating ? input.dtype() : Dtype::kInt64, input.device());
    if (input.device() == Device::kCPU) {
        SumCPU(input.data_ptr(), result.data_ptr(), input.dtype(), input.shape(),
               input.stride(), input.numel(), dim);
    } else if (input.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        SumCUDA(input.data_ptr(), result.data_ptr(), input.dtype(), input.shape_cuda(),
                input.stride_cuda(), input.dim(), result.numel(), dim);
#else
        throw std::logic_error("sum: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("sum: unsupported device");
    }
    
    if(EnableGrad() && CheckGradDtype(input.dtype())) {
        auto node = std::make_shared<SumNode>();
        node->inputs = {input};
        node->dim = dim;
        node->input_shape = input.shape();
        result.set_grad_node(node);
    }
    
    return result;
}

Tensor Tensor::sum() const {
    return mtensor::sum(*this);
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    return mtensor::sum(*this, dim, keepdim);
}

Tensor mean(const Tensor& input) {
    CheckFloatingDtype(input.dtype(), "mean: input must have floating dtype");
    Tensor total = sum(input);
    Tensor scale = full(total.shape(), 1.0f / static_cast<float>(input.numel()),
                        input.dtype(), input.device());
    return mul(total, scale);
}

Tensor mean(const Tensor& input, int dim, bool keepdim) {
    CheckFloatingDtype(input.dtype(), "mean: input must have floating dtype");
    Tensor total = sum(input, dim, keepdim);
    if (dim < 0) dim += input.dim();
    Tensor scale = full(total.shape(), 1.0f / static_cast<float>(input.size(dim)),
                        input.dtype(), input.device());
    return mul(total, scale);
}

Tensor Tensor::mean() const {
    return mtensor::mean(*this);
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    return mtensor::mean(*this, dim, keepdim);
}


std::vector<Tensor> SumNode::backward(const Tensor& output_grad) {
    const auto& input_shape = inputs[0].shape();

    if (dim == -1) {
        return {output_grad.broadcast(input_shape)};
    }

    auto grad_shape = input_shape;
    grad_shape[dim] = 1;

    Tensor grad = output_grad.reshape(grad_shape);
    return {grad.broadcast(input_shape)};
}

}  // namespace mtensor
