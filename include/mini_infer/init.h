#pragma once

#include <cstddef>
#include <functional>
#include <random>

namespace mtensor::weight_init {

// 在层分配 Tensor 之前，原地填充连续的主机 FP32 缓冲区。
void normal_(float* values, std::size_t count, std::mt19937& rng,
             float mean = 0.0f, float stddev = 1.0f);
void uniform_(float* values, std::size_t count, std::mt19937& rng,
              float low = 0.0f, float high = 1.0f);

enum class FanMode { kFanIn, kFanOut };

// 权重形状为 [fan_out, fan_in]；a 是 Leaky ReLU 负半轴斜率，a=0 即 ReLU。
void kaiming_uniform_(float* values, std::mt19937& rng, int fan_in, int fan_out,
                      double a = 0.0, FanMode mode = FanMode::kFanIn);
void kaiming_normal_(float* values, std::mt19937& rng, int fan_in, int fan_out,
                     double a = 0.0, FanMode mode = FanMode::kFanIn);

// 调用方持有 rng；层在构造期间立即消费初始化器。
std::function<void(float*, std::size_t)> make_normal_initializer(std::mt19937& rng, float stddev);

}  // namespace mtensor::weight_init
