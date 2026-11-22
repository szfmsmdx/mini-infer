#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <stdexcept>

#include "init/weight_init.h"

namespace mtensor::weight_init {
namespace {

double kaiming_std(int fan_in, int fan_out, double a, FanMode mode) {
    if (fan_in <= 0 || fan_out <= 0 || !std::isfinite(a)) {
        throw std::invalid_argument("Kaiming initialization requires positive fan dimensions and a finite slope");
    }
    const int fan = mode == FanMode::kFanIn ? fan_in : fan_out;
    const double gain = std::sqrt(2.0) / std::hypot(1.0, a);
    return gain / std::sqrt(static_cast<double>(fan));
}

}  // namespace

void normal_(float* values, std::size_t count, std::mt19937& rng, float mean, float stddev) {
    if (!std::isfinite(stddev) || stddev < 0.0f) {
        throw std::invalid_argument("normal_: standard deviation must be finite and nonnegative");
    }
    if (stddev == 0.0f) {
        for (std::size_t i = 0; i < count; ++i) values[i] = mean;
        return;
    }
    std::normal_distribution<float> normal(mean, stddev);
    for (std::size_t i = 0; i < count; ++i) values[i] = normal(rng);
}

void uniform_(float* values, std::size_t count, std::mt19937& rng, float low, float high) {
    if (!(low <= high)) {
        throw std::invalid_argument("uniform_: low must not exceed high");
    }
    std::uniform_real_distribution<float> uniform(low, high);
    for (std::size_t i = 0; i < count; ++i) values[i] = uniform(rng);
}

void kaiming_uniform_(float* values, std::mt19937& rng, int fan_in, int fan_out,
                      double a, FanMode mode) {
    const double stddev = kaiming_std(fan_in, fan_out, a, mode);
    const float bound = static_cast<float>(std::sqrt(3.0) * stddev);
    uniform_(values, static_cast<std::size_t>(fan_out) * fan_in, rng, -bound, bound);
}

void kaiming_normal_(float* values, std::mt19937& rng, int fan_in, int fan_out,
                     double a, FanMode mode) {
    const auto stddev = static_cast<float>(kaiming_std(fan_in, fan_out, a, mode));
    normal_(values, static_cast<std::size_t>(fan_out) * fan_in, rng, 0.0f, stddev);
}

std::function<void(float*, std::size_t)> make_normal_initializer(std::mt19937& rng, float stddev) {
    return [&rng, stddev](float* values, std::size_t count) {
        normal_(values, count, rng, 0.0f, stddev);
    };
}

}  // namespace mtensor::weight_init
