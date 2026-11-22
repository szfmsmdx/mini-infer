#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <numeric>

namespace benchmark {

template <typename Run>
double TimeMs(Run run) {
    const auto start = std::chrono::steady_clock::now();
    run();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

template <typename Samples>
double Mean(const Samples& samples) {
    return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
}

template <typename Samples>
double Median(Samples samples) {
    std::sort(samples.begin(), samples.end());
    return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}

template <typename RunMini, typename RunGgml>
void Warmup(RunMini run_mini, RunGgml run_ggml, int count) {
    for (int i = 0; i < count; ++i) {
        run_mini();
        run_ggml();
    }
}

// 两组 samples 由调用方按相同采样次数创建，回调定义各自的计时边界。
template <typename RunMini, typename RunGgml, typename Samples>
void MeasureAlternating(RunMini run_mini, RunGgml run_ggml,
                        Samples& mini, Samples& ggml) {
    for (std::size_t i = 0; i < mini.size(); ++i) {
        if (i % 2 == 0) {
            mini[i] = TimeMs(run_mini);
            ggml[i] = TimeMs(run_ggml);
        } else {
            ggml[i] = TimeMs(run_ggml);
            mini[i] = TimeMs(run_mini);
        }
    }
}

}  // namespace benchmark
