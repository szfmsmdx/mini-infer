#include <memory>
#include <stdexcept>
#include <vector>

#include "ops/reduction.h"
#include "autograd/autograd.h"
#include "autograd/backward_nodes/reduction.h"
#include "ops/backend.h"

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

}  // namespace mtensor
