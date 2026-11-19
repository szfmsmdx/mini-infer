#pragma once

#include <cstdint>

#include "mini_infer/autograd.h"

namespace mtensor {

class MatmulNode : public AutogradNode {
public:
    std::uint64_t input_versions[2] = {};
    std::vector<Tensor> backward(const Tensor& output_grad) override;
};

}  // namespace mtensor
