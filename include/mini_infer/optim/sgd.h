#pragma once

#include <vector>

#include "mini_infer/optim/optimizer.h"

namespace mtensor {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> parameters, float learning_rate, float momentum = 0.0f);
    void step() override;

private:
    float momentum_;
    std::vector<Tensor> momentum_buffers_;
};

}  // namespace mtensor
