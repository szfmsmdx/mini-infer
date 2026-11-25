#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
        SumCUDA(input.data_ptr(), result.data_ptr(), input.dtype(), PackLayout(input), input.numel());
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

    auto output_shape = input.shape().to_vector();
    if (keepdim) output_shape[dim] = 1;
    else output_shape.erase(output_shape.begin() + dim);
    
    Tensor result(output_shape, floating ? input.dtype() : Dtype::kInt64, input.device());
    if (input.device() == Device::kCPU) {
        SumCPU(input.data_ptr(), result.data_ptr(), input.dtype(), input.shape(),
               input.stride(), input.numel(), dim);
    } else if (input.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        SumCUDA(input.data_ptr(), result.data_ptr(), input.dtype(), PackLayout(input), result.numel(), dim);
#else
        throw std::logic_error("sum: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("sum: unsupported device");
    }
    
    if (EnableGrad() && floating) {
        auto node = std::make_shared<SumNode>();
        node->inputs = {input};
        node->dim = dim;
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

namespace {

class ExtremaNode : public AutogradNode {
public:
    Tensor extrema;
    std::size_t inner_size = 1;
    std::uint64_t input_version = 0;
    std::uint64_t output_version = 0;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        const Tensor& input = inputs[0];
        CheckVersion(input, input_version, "extrema: input modified after forward");
        CheckVersion(extrema, output_version, "extrema: output modified after forward");
        const Tensor grad = output_grad.contiguous();
        Tensor input_grad(input.shape(), input.dtype(), input.device());
        if (input.device() == Device::kCPU) {
            ExtremaGradCPU(input, extrema, grad, input_grad, inner_size);
        } else {
#ifdef ENABLE_CUDA
            ExtremaGradCUDA(input, extrema, grad, input_grad, inner_size);
#else
            throw std::logic_error("extrema: CUDA support is disabled");
#endif
        }
        return {input_grad};
    }
};

Tensor ReduceExtrema(const Tensor& input, std::optional<int> dim, bool keepdim, bool maximum) {
    CheckNotEmpty(input, "max/min: input must not be empty");
    std::vector<int> output_shape;
    std::size_t inner_size = 1;
    if (dim) {
        int axis = *dim;
        if (axis < 0) axis += input.dim();
        CheckDim(axis, input.shape().size(), "max/min: dim out of range");
        for (int i = axis + 1; i < input.dim(); ++i) inner_size *= input.size(i);

        output_shape = input.shape().to_vector();
        if (keepdim) output_shape[axis] = 1;
        else output_shape.erase(output_shape.begin() + axis);
    }

    Tensor output(output_shape, input.dtype(), input.device());
    if (input.device() == Device::kCPU) {
        ExtremaCPU(input, output, inner_size, maximum);
    } else if (input.device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        ExtremaCUDA(input, output, inner_size, maximum);
#else
        throw std::logic_error("max/min: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("max/min: unsupported device");
    }

    if (EnableGrad() && CheckGradDtype(input.dtype())) {
        auto node = std::make_shared<ExtremaNode>();
        node->inputs = {input};
        node->extrema = output.detach();
        node->inner_size = inner_size;
        node->input_version = input.version();
        node->output_version = output.version();
        output.set_grad_node(node);
    }
    return output;
}

}  // namespace

Tensor max(const Tensor& input) {
    return ReduceExtrema(input, std::nullopt, false, true);
}

Tensor max(const Tensor& input, int dim, bool keepdim) {
    return ReduceExtrema(input, dim, keepdim, true);
}

Tensor min(const Tensor& input) {
    return ReduceExtrema(input, std::nullopt, false, false);
}

Tensor min(const Tensor& input, int dim, bool keepdim) {
    return ReduceExtrema(input, dim, keepdim, false);
}

Tensor Tensor::max() const { return mtensor::max(*this); }
Tensor Tensor::max(int dim, bool keepdim) const { return mtensor::max(*this, dim, keepdim); }
Tensor Tensor::min() const { return mtensor::min(*this); }
Tensor Tensor::min(int dim, bool keepdim) const { return mtensor::min(*this, dim, keepdim); }

std::vector<Tensor> SumNode::backward(const Tensor& output_grad) {
    const auto& input_shape = inputs[0].shape();

    if (dim == -1) {
        return {output_grad.broadcast(input_shape)};
    }

    auto grad_shape = input_shape.to_vector();
    grad_shape[dim] = 1;

    Tensor grad = output_grad.reshape(grad_shape);
    return {grad.broadcast(input_shape)};
}

}  // namespace mtensor
