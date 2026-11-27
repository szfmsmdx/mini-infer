#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "mini_infer/ops/reduction.h"
#include "backend/dispatch.h"
#include "tensor/check.h"
#include "mini_infer/autograd.h"
#include "ops/reduction/nodes.h"
#include "ops/reduction/kernels.h"

namespace mtensor {

Tensor sum(const Tensor& input) {
    CheckNotEmpty(input, "sum: input must not be empty");
    const bool floating = CheckGradDtype(input.dtype());
    Tensor result(/*shape=*/std::vector<int>{},
                  /*dtype=*/floating ? input.dtype() : Dtype::kInt64,
                  /*device=*/input.device());
    GetKernelTable().sum_all.call(input.device(), "sum", input, result);

    if (floating && ShouldRecordGrad({&input}, result.dtype())) {
        auto node = std::make_shared<SumNode>();
        node->record_inputs({input});
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
    GetKernelTable().sum_dim.call(input.device(), "sum", input, dim, result);
    
    if (floating && ShouldRecordGrad({&input}, result.dtype())) {
        auto node = std::make_shared<SumNode>();
        node->record_inputs({input});
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

namespace {

class MeanNode : public AutogradNode {
public:
    int dim = -1;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        NogradNode guard;
        const Tensor& input = inputs[0];
        Tensor dx;
        GetKernelTable().mean_backward.call(input.device(), "mean backward", input, output_grad, dim, dx);
        return {dx};
    }
};

Tensor ReduceMean(const Tensor& input, std::optional<int> dim, bool keepdim) {
    CheckFloatingDtype(input.dtype(), "mean: input must have floating dtype");
    CheckNotEmpty(input, "mean: input must not be empty");
    int axis = -1;
    std::vector<int> output_shape;
    if (dim) {
        axis = *dim;
        if (axis < 0) axis += input.dim();
        CheckDim(axis, input.shape().size(), "mean: dim out of range");
        output_shape = input.shape().to_vector();
        if (keepdim) output_shape[axis] = 1;
        else output_shape.erase(output_shape.begin() + axis);
    }

    Tensor output(output_shape, input.dtype(), input.device());
    {
        NogradNode guard;
        GetKernelTable().mean.call(input.device(), "mean", input, axis, keepdim, output);
    }
    if (ShouldRecordGrad({&input}, output.dtype())) {
        auto node = std::make_shared<MeanNode>();
        node->record_inputs({input});
        node->dim = axis;
        output.set_grad_node(node);
    }
    return output;
}

}  // namespace

Tensor mean(const Tensor& input) {
    return ReduceMean(input, std::nullopt, false);
}

Tensor mean(const Tensor& input, int dim, bool keepdim) {
    return ReduceMean(input, dim, keepdim);
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
    SavedTensor saved_input;
    SavedTensor saved_extrema;
    std::size_t inner_size = 1;

    std::vector<Tensor> backward(const Tensor& output_grad) override {
        const Tensor input = saved_input.unpack("extrema input");
        const Tensor extrema = saved_extrema.unpack("extrema output");
        const Tensor grad = output_grad.contiguous();
        Tensor input_grad(input.shape(), input.dtype(), input.device());
        GetKernelTable().extrema_backward.call(
            input.device(), "extrema backward", input, extrema, grad, input_grad, inner_size);
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
    GetKernelTable().extrema.call(
        input.device(), "max/min", input, output, inner_size, maximum);

    if (ShouldRecordGrad({&input}, output.dtype())) {
        auto node = std::make_shared<ExtremaNode>();
        node->record_inputs({input});
        node->saved_input = SavedTensor(input);
        node->saved_extrema = SavedTensor(output, /*is_output=*/true);
        node->inner_size = inner_size;
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
