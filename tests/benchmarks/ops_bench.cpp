#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef OPS_BENCH_CUDA
#include <cuda_runtime.h>
#include <ggml-cuda.h>
#endif

#include "bench_common.h"
#include "mini_infer/autograd.h"
#include "mini_infer/modules/linear.h"
#include "mini_infer/ops.h"
#ifdef OPS_BENCH_CUDA
#include "runtime/cuda_check.h"
#endif

namespace {
using namespace mtensor;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr float kEps = 1e-5f;
#ifdef OPS_BENCH_CUDA
constexpr Device kDevice = Device::kCUDA;
#else
constexpr Device kDevice = Device::kCPU;
#endif

enum class Op {
    Add, Sub, Mul, Div, Neg, Sqrt, Rsqrt, Exp, Sigmoid, Silu,
    SumAll, SumLast, MeanAll, MeanLast, Matmul, Linear, RMSNorm
};

struct Operator {
    Op op;
    const char* name;
};

constexpr Operator kOperators[] = {
    {Op::Add, "add"}, {Op::Sub, "sub"}, {Op::Mul, "mul"}, {Op::Div, "div"},
    {Op::Neg, "neg"}, {Op::Sqrt, "sqrt"}, {Op::Rsqrt, "rsqrt"},
    {Op::Exp, "exp"}, {Op::Sigmoid, "sigmoid"}, {Op::Silu, "silu"},
    {Op::SumAll, "sum_all"}, {Op::SumLast, "sum_last"},
    {Op::MeanAll, "mean_all"}, {Op::MeanLast, "mean_last"},
    {Op::Matmul, "matmul"}, {Op::Linear, "linear"}, {Op::RMSNorm, "rmsnorm"}
};

struct Options {
    int samples = 10;
    int warmups = 3;
    int rows = 0;
    int cols = 1024;
    int ggml_threads = GGML_DEFAULT_N_THREADS;
    std::string op = "all";
};

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("missing option value");
        const std::string option = argv[i];
        const std::string value = argv[i + 1];
        if (option == "--op") {
            options.op = value;
            continue;
        }
        std::size_t consumed = 0;
        const int number = std::stoi(value, &consumed);
        if (consumed != value.size() || number <= 0) {
            throw std::invalid_argument("numeric options must be positive integers");
        }
        if (option == "--samples") options.samples = number;
        else if (option == "--warmups") options.warmups = number;
        else if (option == "--rows") options.rows = number;
        else if (option == "--cols") options.cols = number;
        else if (option == "--ggml-threads") options.ggml_threads = number;
        else throw std::invalid_argument("unknown option: " + option);
    }
    if (options.op != "all" &&
        std::none_of(std::begin(kOperators), std::end(kOperators),
                     [&](const Operator& op) { return options.op == op.name; })) {
        throw std::invalid_argument("unknown operator: " + options.op);
    }
    return options;
}

struct Inputs {
    Tensor x, b, weight, bias;
    std::optional<Linear> linear;
    ggml_tensor* gx = nullptr;
    ggml_tensor* gb = nullptr;
    ggml_tensor* gweight = nullptr;
    ggml_tensor* gbias = nullptr;
};

Tensor RunMini(Op op, Inputs& inputs) {
    const auto& x = inputs.x;
    const auto& b = inputs.b;
    switch (op) {
        case Op::Add: return add(x, b);
        case Op::Sub: return sub(x, b);
        case Op::Mul: return mul(x, b);
        case Op::Div: return div(x, b);
        case Op::Neg: return neg(x);
        case Op::Sqrt: return sqrt(x);
        case Op::Rsqrt: return rsqrt(x);
        case Op::Exp: return exp(x);
        case Op::Sigmoid: return sigmoid(x);
        case Op::Silu: return silu(x);
        case Op::SumAll: return sum(x);
        case Op::SumLast: return sum(x, -1, true);
        case Op::MeanAll: return mean(x);
        case Op::MeanLast: return mean(x, -1, true);
        case Op::Matmul: return matmul(x, b);
        case Op::Linear: return (*inputs.linear)(x);
        case Op::RMSNorm: return rms_norm(x, inputs.weight, kEps);
    }
    throw std::logic_error("unhandled mini-infer operator");
}

ggml_tensor* BuildGgml(ggml_context* ctx, Op op, const Inputs& inputs) {
    auto* x = inputs.gx;
    auto* b = inputs.gb;
    switch (op) {
        case Op::Add: return ggml_add(ctx, x, b);
        case Op::Sub: return ggml_sub(ctx, x, b);
        case Op::Mul: return ggml_mul(ctx, x, b);
        case Op::Div: return ggml_div(ctx, x, b);
        case Op::Neg: return ggml_neg(ctx, x);
        case Op::Sqrt: return ggml_sqrt(ctx, x);
        case Op::Rsqrt: return ggml_div(ctx, b, ggml_sqrt(ctx, x));
        case Op::Exp: return ggml_exp(ctx, x);
        case Op::Sigmoid: return ggml_sigmoid(ctx, x);
        case Op::Silu: return ggml_silu(ctx, x);
        case Op::SumAll: return ggml_sum(ctx, x);
        case Op::SumLast: return ggml_sum_rows(ctx, x);
        case Op::MeanAll: return ggml_mean(ctx, ggml_reshape_1d(ctx, x, ggml_nelements(x)));
        case Op::MeanLast: return ggml_mean(ctx, x);
        case Op::Matmul: return ggml_mul_mat(ctx, inputs.gweight, x);
        case Op::Linear: return ggml_add(ctx, ggml_mul_mat(ctx, inputs.gweight, x), inputs.gbias);
        case Op::RMSNorm: return ggml_mul(ctx, ggml_rms_norm(ctx, x, kEps), inputs.gweight);
    }
    throw std::logic_error("unhandled ggml operator");
}

// 每次完整调用各自持有图、输出和临时存储，退出作用域时释放。
struct GgmlCall {
    Context context{nullptr, ggml_free};
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor* output = nullptr;
    ggml_cgraph* graph = nullptr;

    GgmlCall(ggml_backend_t backend, Op op, const Inputs& inputs) {
        const std::size_t bytes = 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false);
        context.reset(ggml_init({bytes, nullptr, true}));
        if (!context) throw std::runtime_error("ggml output context allocation failed");
        output = BuildGgml(context.get(), op, inputs);
        graph = ggml_new_graph_custom(context.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) throw std::runtime_error("ggml output buffer allocation failed");
    }

    void Compute(ggml_backend_t backend) {
        // 同步图接口内部已等待后端完成。
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml graph compute failed");
        }
    }
};

void SynchronizeMini() {
#ifdef OPS_BENCH_CUDA
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
#endif
}

double CheckOutput(const Tensor& actual, const GgmlCall& reference) {
    if (actual.dtype() != Dtype::kFloat32 || reference.output->type != GGML_TYPE_F32 ||
        actual.numel() != static_cast<std::size_t>(ggml_nelements(reference.output))) {
        throw std::runtime_error("output dtype or element count differs");
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        const int expected = dim < actual.dim() ? actual.size(actual.dim() - 1 - dim) : 1;
        if (reference.output->ne[dim] != expected) throw std::runtime_error("output shape differs");
    }
    const Tensor host = actual.to(Device::kCPU);
    const auto* values = static_cast<const float*>(host.data_ptr());
    std::vector<float> expected(actual.numel());
    ggml_backend_tensor_get(reference.output, expected.data(), 0, expected.size() * sizeof(float));
    double max_error = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(static_cast<double>(values[i]) - expected[i]);
        if (!std::isfinite(values[i]) || !std::isfinite(expected[i]) ||
            error > 2e-4 + 2e-5 * std::abs(expected[i])) {
            throw std::runtime_error("output mismatch at element " + std::to_string(i));
        }
        max_error = std::max(max_error, error);
    }
    return max_error;
}

void RunCase(ggml_backend_t backend, const Operator& op, int rows, const Options& options) {
    const int cols = options.cols;
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    const bool binary = op.op == Op::Add || op.op == Op::Sub || op.op == Op::Mul || op.op == Op::Div;
    const bool matrix = op.op == Op::Matmul || op.op == Op::Linear;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> distribution(-8, 8);
    std::vector<float> x(count), b, weight, bias;
    for (float& value : x) value = 1.0f + distribution(rng) / 16.0f;
    if (binary || op.op == Op::Rsqrt) {
        b.resize(count, 1.0f);
        if (binary) for (float& value : b) value = 1.0f + distribution(rng) / 16.0f;
    }
    if (matrix || op.op == Op::RMSNorm) {
        weight.resize(matrix ? static_cast<std::size_t>(cols) * cols : cols);
        for (float& value : weight) value = distribution(rng) / 16.0f;
    }
    if (op.op == Op::Linear) bias.assign(cols, 0.125f);

    Inputs inputs;
    inputs.x = Tensor(x.data(), {rows, cols}, Dtype::kFloat32, kDevice);
    Context input_context(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    if (!input_context) throw std::runtime_error("ggml input context allocation failed");
    inputs.gx = ggml_new_tensor_2d(input_context.get(), GGML_TYPE_F32, cols, rows);
    if (!b.empty()) {
        inputs.b = Tensor(b.data(), {rows, cols}, Dtype::kFloat32, kDevice);
        inputs.gb = ggml_new_tensor_2d(input_context.get(), GGML_TYPE_F32, cols, rows);
    }
    if (!weight.empty()) {
        const std::vector<int> shape = matrix ? std::vector<int>{cols, cols} : std::vector<int>{cols};
        inputs.weight = Tensor(weight.data(), shape, Dtype::kFloat32, kDevice);
        inputs.gweight = ggml_new_tensor_2d(input_context.get(), GGML_TYPE_F32, cols, matrix ? cols : 1);
    }
    if (!bias.empty()) {
        inputs.bias = Tensor(bias.data(), {cols}, Dtype::kFloat32, kDevice);
        inputs.gbias = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_F32, cols);
        inputs.linear.emplace(inputs.weight, inputs.bias);
    }
    if (op.op == Op::Matmul) inputs.b = inputs.weight.transpose(0, 1);
    Buffer input_buffer(ggml_backend_alloc_ctx_tensors(input_context.get(), backend), ggml_backend_buffer_free);
    if (!input_buffer) throw std::runtime_error("ggml input buffer allocation failed");
    ggml_backend_tensor_set(inputs.gx, x.data(), 0, x.size() * sizeof(float));
    if (inputs.gb) ggml_backend_tensor_set(inputs.gb, b.data(), 0, b.size() * sizeof(float));
    if (inputs.gweight) ggml_backend_tensor_set(inputs.gweight, weight.data(), 0, weight.size() * sizeof(float));
    if (inputs.gbias) ggml_backend_tensor_set(inputs.gbias, bias.data(), 0, bias.size() * sizeof(float));
    SynchronizeMini();
    ggml_backend_synchronize(backend);

    double max_error;
    {
        GgmlCall reference(backend, op.op, inputs);
        for (int i = 0; i < ggml_graph_n_nodes(reference.graph); ++i) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(reference.graph, i))) {
                std::cout << op.name << ',' << rows << ',' << cols
                          << ",float32,,,,,,,unsupported_ggml" << std::endl;
                return;
            }
        }
        reference.Compute(backend);
        const Tensor actual = RunMini(op.op, inputs);
        SynchronizeMini();
        max_error = CheckOutput(actual, reference);
    }

    // 两边均计入输出/临时分配和释放；输入与模型参数在计时外准备。
    const auto run_mini = [&] {
        const Tensor output = RunMini(op.op, inputs);
        SynchronizeMini();
    };
    const auto run_ggml = [&] {
        GgmlCall call(backend, op.op, inputs);
        call.Compute(backend);
    };
    benchmark::Warmup(run_mini, run_ggml, options.warmups);
    std::vector<double> mini(options.samples), ggml(options.samples);
    benchmark::MeasureAlternating(run_mini, run_ggml, mini, ggml);
    // 再核对一次，防止重复调用改变输入或留下错误结果。
    {
        GgmlCall reference(backend, op.op, inputs);
        reference.Compute(backend);
        const Tensor actual = RunMini(op.op, inputs);
        SynchronizeMini();
        max_error = std::max(max_error, CheckOutput(actual, reference));
    }
    std::cout << op.name << ',' << rows << ',' << cols << ",float32,"
              << benchmark::Mean(mini) << ',' << benchmark::Mean(ggml) << ','
              << benchmark::Median(mini) << ',' << benchmark::Median(ggml) << ','
              << benchmark::Mean(mini) / benchmark::Mean(ggml) << ',' << max_error << ",pass" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(nullptr, ggml_backend_free);
#ifdef OPS_BENCH_CUDA
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp gpu{};
        CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
        backend.reset(ggml_backend_cuda_init(0));
        std::cout << "# device=" << gpu.name << " SM" << gpu.major << gpu.minor << '\n';
#else
        backend.reset(ggml_backend_cpu_init());
        if (backend) ggml_backend_cpu_set_n_threads(backend.get(), options.ggml_threads);
        std::cout << "# mini_threads=1 ggml_threads=" << options.ggml_threads
                  << " ggml_AVX2=" << ggml_cpu_has_avx2() << " ggml_FMA=" << ggml_cpu_has_fma() << '\n';
#endif
        if (!backend) throw std::runtime_error("ggml backend initialization failed");
        NogradNode guard;
        std::cout << std::fixed << std::setprecision(6)
                  << "# backend=" << ggml_backend_dev_description(ggml_backend_get_device(backend.get())) << '\n'
                  << "# scope=full_forward_call inputs_prepared_outside_timing dtype=float32\n"
                  << "# includes=output_and_temporary_allocation,ggml_graph_build,compute,sync,cleanup\n"
                  << "# ggml_revision=" << BENCH_GGML_REVISION << '\n'
                  << "# mini_revision=" << BENCH_MINI_REVISION << " config=" << BENCH_BUILD_CONFIG << '\n'
#ifdef _MSC_VER
                  << "# compiler=MSVC " << _MSC_VER << '\n'
#else
                  << "# compiler=" << __VERSION__ << '\n'
#endif
                  << "# warmups=" << options.warmups << " samples=" << options.samples
                  << " alternating_order=true seed=42\n"
                  << "op,rows,cols,dtype,mini_mean_ms,ggml_mean_ms,mini_median_ms,ggml_median_ms,"
                     "mini_over_ggml,max_abs_error,status" << std::endl;
        const std::vector<int> rows = options.rows ? std::vector<int>{options.rows} : std::vector<int>{1, 32, 128};
        for (const auto& op : kOperators) {
            if (options.op != "all" && options.op != op.name) continue;
            for (int count : rows) RunCase(backend.get(), op, count, options);
        }
    } catch (const std::exception& error) {
        std::cerr << "ops benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
