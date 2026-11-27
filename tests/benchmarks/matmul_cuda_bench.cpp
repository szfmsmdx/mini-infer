#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <cuda_runtime.h>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "bench_common.h"
#include "ops/matmul/kernels.h"
#include "mini_infer/tensor/tensor.h"
#include "runtime/cuda_check.h"

namespace {
constexpr int kSamples = 10;
constexpr int kWarmups = 3;
using Samples = std::array<double, kSamples>;

void RunCase(ggml_backend_t backend, int m, int k, int n) {
    using namespace mtensor;
    const auto input_elements = static_cast<std::size_t>(m) * k;
    const auto weight_elements = static_cast<std::size_t>(n) * k;
    const auto output_elements = static_cast<std::size_t>(m) * n;
    std::vector<float> input(input_elements), weight(weight_elements);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> values(-8, 8);
    // Binary fractions keep dot products exact in FP32 for the tested K values.
    for (auto& value : input) value = values(rng) / 16.0f;
    for (auto& value : weight) value = values(rng) / 16.0f;

    const Tensor a(input.data(), {m, k}, Dtype::kFloat32, Device::kCUDA);
    const Tensor b = Tensor(weight.data(), {n, k}, Dtype::kFloat32, Device::kCUDA)
                         .transpose(0, 1);
    Tensor c({m, n}, Dtype::kFloat32, Device::kCUDA);

    const std::size_t metadata_bytes = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({metadata_bytes, nullptr, true}), ggml_free);
    if (!ctx) throw std::runtime_error("ggml_init failed");
    auto* lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    auto* rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, n);
    auto* result = ggml_mul_mat(ctx.get(), rhs, lhs);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    if (!ggml_backend_supports_op(backend, result)) {
        throw std::runtime_error("ggml CUDA does not support this matmul");
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("ggml CUDA buffer allocation failed");
    ggml_backend_tensor_set(lhs, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, weight.data(), 0, weight.size() * sizeof(float));

    const auto run_mini = [&] {
        MatmulCUDA(a.data_ptr(), b.data_ptr(), c.data_ptr(), Dtype::kFloat32,
                   m, k, n, a.stride(), b.stride());
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    };
    const auto run_ggml = [&] {
        // The synchronous graph API waits on the CUDA backend's stream.
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml CUDA graph compute failed");
        }
    };
    benchmark::Warmup(run_mini, run_ggml, kWarmups);
    const Tensor host_c = c.to(Device::kCPU);
    const auto* actual = static_cast<const float*>(host_c.data_ptr());
    std::vector<float> reference(output_elements);
    ggml_backend_tensor_get(result, reference.data(), 0, reference.size() * sizeof(float));
    if (!std::equal(reference.begin(), reference.end(), actual)) {
        throw std::runtime_error("mini-infer and ggml CUDA outputs differ");
    }

    Samples mini{}, ggml{};
    benchmark::MeasureAlternating(run_mini, run_ggml, mini, ggml);
    const double mini_mean = benchmark::Mean(mini);
    const double ggml_mean = benchmark::Mean(ggml);
    const double flops = 2.0 * m * k * n;
    std::cout << m << '\t' << k << '\t' << n << '\t'
              << mini_mean << '\t' << ggml_mean << '\t'
              << benchmark::Median(mini) << '\t' << benchmark::Median(ggml) << '\t'
              << flops / (mini_mean * 1e6) << '\t' << flops / (ggml_mean * 1e6) << '\t'
              << mini_mean / ggml_mean << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
    const bool model_cases = argc == 2 && std::string_view(argv[1]) == "--model";
    if (argc != 1 && !model_cases) {
        std::cerr << "Usage: matmul_cuda_bench [--model]\n";
        return 1;
    }
    try {
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_cuda_init(0), ggml_backend_free);
        if (!backend) throw std::runtime_error("ggml CUDA backend initialization failed");
        std::cout << std::fixed << std::setprecision(4)
                  << device.name << " (SM" << device.major << device.minor << "), FP32, weight.T, "
                  << kWarmups << " warmups, " << kSamples << " timed runs per backend\n"
                  << "M\tK\tN\tmini mean ms\tggml mean ms\tmini median ms\tggml median ms\t"
                     "mini GFLOP/s\tggml GFLOP/s\tmini/ggml" << std::endl;
        if (model_cases) {
            for (int m : {1, 2, 4, 8, 32, 128, 512}) RunCase(backend.get(), m, 4096, 4096);
            for (int m : {1, 32, 512}) {
                RunCase(backend.get(), m, 4096, 14336);
                RunCase(backend.get(), m, 14336, 4096);
            }
        } else {
            for (int m : {1, 32, 128}) RunCase(backend.get(), m, 1024, 1024);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
