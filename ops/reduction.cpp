#include <memory>
#include <stdexcept>
#include <vector>

#include "ops/reduction.h"
#include "autograd/autograd.h"
#include "autograd/backward_nodes/reduction.h"
#include "ops/backend.h"
#include "ops/elementwise.h"

namespace mtensor {

Tensor sum(const Tensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("sum: input must not be empty");
    }
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
    if (input.empty()) {
        throw std::invalid_argument("sum: input must not be empty");
    }
    if (dim < 0) dim += input.dim();
    if (dim < 0 || dim >= input.dim()) {
        throw std::out_of_range("sum: dim out of range");
    }

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
    if (!CheckGradDtype(input.dtype())) {
        throw std::invalid_argument("mean: input must have floating dtype");
    }
    Tensor total = sum(input);
    Tensor scale = full(total.shape(), 1.0f / static_cast<float>(input.numel()),
                        input.dtype(), input.device());
    return mul(total, scale);
}

Tensor mean(const Tensor& input, int dim, bool keepdim) {
    if (!CheckGradDtype(input.dtype())) {
        throw std::invalid_argument("mean: input must have floating dtype");
    }
    Tensor total = sum(input, dim, keepdim);
    if (dim < 0) dim += input.dim();
    Tensor scale = full(total.shape(), 1.0f / static_cast<float>(input.size(dim)),
                        input.dtype(), input.device());
    return mul(total, scale);
}

}  // namespace mtensor
