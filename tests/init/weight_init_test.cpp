#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "mini_infer/init.h"

namespace {
using namespace mtensor::weight_init;

void CheckKaimingUniform() {
    constexpr int fan_in = 8;
    constexpr int fan_out = 2;
    std::array<float, fan_in * fan_out> values;
    std::mt19937 actual_rng(42), reference_rng(42);
    kaiming_uniform_(values.data(), actual_rng, fan_in, fan_out, std::sqrt(5.0));

    const float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
    std::uniform_real_distribution<float> expected(-bound, bound);
    for (float value : values) {
        if (!(std::abs(value - expected(reference_rng)) <= 1e-6f)) {
            throw std::runtime_error("Kaiming uniform did not use the Linear fan_in bound");
        }
    }
}

void CheckKaimingNormal() {
    constexpr int fan_in = 8;
    constexpr int fan_out = 2;
    std::array<float, fan_in * fan_out> values;
    std::mt19937 actual_rng(42), reference_rng(42);
    kaiming_normal_(values.data(), actual_rng, fan_in, fan_out, 0.0, FanMode::kFanOut);

    // ReLU 的 gain 是 sqrt(2)，fan_out 为 2 时标准差为 1。
    std::normal_distribution<float> expected(0.0f, 1.0f);
    for (float value : values) {
        if (value != expected(reference_rng)) {
            throw std::runtime_error("Kaiming normal did not use the ReLU fan_out scale");
        }
    }
}

}  // namespace

int main() {
    try {
        CheckKaimingUniform();
        CheckKaimingNormal();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
