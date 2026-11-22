#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <ggml.h>
#include <ggml-cpu.h>

#include "bench_common.h"
#include "ops/matmul/kernels.h"

namespace {
constexpr int kSamples = 10;
constexpr int kWarmups = 3;
using Samples = std::array<double, kSamples>;

void RunCase(int m, int k, int n, bool transposed_weight) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> a(m * k), b(k * n), c(m * n);
    for (auto& value : a) value = distribution(rng);
    // Generate the same logical B for both layouts.
    for (int p = 0; p < k; ++p) {
        for (int j = 0; j < n; ++j) {
            b[transposed_weight ? j * k + p : p * n + j] = distribution(rng);
        }
    }

    const std::vector<std::size_t> a_stride{static_cast<std::size_t>(k), 1};
    const std::vector<std::size_t> b_stride = transposed_weight
        ? std::vector<std::size_t>{1, static_cast<std::size_t>(k)}
        : std::vector<std::size_t>{static_cast<std::size_t>(n), 1};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({32 * 1024 * 1024, nullptr, false}), ggml_free);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    auto* lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    auto* rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32,
                                   transposed_weight ? k : n, transposed_weight ? n : k);
    std::copy(a.begin(), a.end(), static_cast<float*>(lhs->data));
    std::copy(b.begin(), b.end(), static_cast<float*>(rhs->data));
    // ggml_mul_mat expects the weight's K dimension contiguous.
    if (!transposed_weight) rhs = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), rhs));
    auto* result = ggml_mul_mat(ctx.get(), rhs, lhs);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    auto thread_params = ggml_threadpool_params_default(GGML_DEFAULT_N_THREADS);
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> threadpool(
        ggml_threadpool_new(&thread_params), ggml_threadpool_free);
    if (!threadpool) throw std::runtime_error("ggml_threadpool_new failed");
    auto plan = ggml_graph_plan(graph, 0, threadpool.get());
    std::vector<std::uint8_t> work(plan.work_size);
    plan.work_data = work.data();

    const auto run_mini = [&] {
        mtensor::MatmulCPU(a.data(), b.data(), c.data(), mtensor::Dtype::kFloat32,
                          m, k, n, a_stride, b_stride);
    };
    const auto run_ggml = [&] {
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml graph compute failed");
        }
    };
    benchmark::Warmup(run_mini, run_ggml, kWarmups);
    const auto* reference = static_cast<const float*>(result->data);
    for (std::size_t i = 0; i < c.size(); ++i) {
        if (!(std::abs(c[i] - reference[i]) <= 2e-4f + 1e-5f * std::abs(reference[i]))) {
            throw std::runtime_error("mini-infer and ggml outputs differ");
        }
    }

    Samples mini{}, ggml{};
    benchmark::MeasureAlternating(run_mini, run_ggml, mini, ggml);
    const double mini_mean = benchmark::Mean(mini);
    const double ggml_mean = benchmark::Mean(ggml);
    const double flops = 2.0 * m * k * n;
    std::cout << m << '\t' << k << '\t' << n << '\t'
              << (transposed_weight ? "weight.T" : "row-major") << '\t'
              << mini_mean << '\t' << ggml_mean << '\t'
              << benchmark::Median(mini) << '\t' << benchmark::Median(ggml) << '\t'
              << flops / (mini_mean * 1e6) << '\t' << flops / (ggml_mean * 1e6) << '\t'
              << mini_mean / ggml_mean << '\n';
}
}  // namespace

int main() {
    try {
        std::cout << std::fixed << std::setprecision(4)
                  << "FP32 CPU, mini-infer: 1 thread, ggml: "
                  << GGML_DEFAULT_N_THREADS << " threads (default), "
                  << kWarmups << " warmups, "
                  << kSamples << " timed runs per backend\n"
                  << "M\tK\tN\tB layout\tmini mean ms\tggml mean ms\t"
                     "mini median ms\tggml median ms\tmini GFLOP/s\tggml GFLOP/s\tmini/ggml\n";
        for (int m : {1, 32, 128}) {
            RunCase(m, 1024, 1024, false);
            RunCase(m, 1024, 1024, true);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
