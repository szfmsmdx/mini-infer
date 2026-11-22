#pragma once

#include <cstdint>
#include <vector>

#include "mini_infer/optim/optimizer.h"

namespace mtensor {

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor> parameters, float learning_rate = 1e-3f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step() override;

private:
    struct State {
        Tensor first_moment;
        Tensor second_moment;
        std::uint64_t step = 0;
    };

    float beta1_;
    float beta2_;
    float eps_;
    std::vector<State> states_;
};

}  // namespace mtensor
