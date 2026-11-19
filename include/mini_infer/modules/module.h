#pragma once

#include "mini_infer/tensor.h"

namespace mtensor {

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;

    Tensor operator()(const Tensor& x) { return forward(x); }
};

}  // namespace mtensor
