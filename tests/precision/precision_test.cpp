#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef PRECISION_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/ops.h"

namespace {
using namespace mtensor;

std::ofstream samples_file;
int exported = 0;

struct Precision {
    Dtype dtype;
    int fraction_bits;
    int min_exponent;
    int max_exponent;
};

// 导出原始编码，保留低精度舍入、次正规数和特殊值。
template <typename T>
void WriteTensor(const Tensor& tensor, int offset = 0) {
    const Tensor host = tensor.contiguous().to(Device::kCPU);
    const auto* values = static_cast<const T*>(host.data_ptr());
    samples_file << "{\"shape\":[";
    for (std::size_t i = 0; i < tensor.shape().size(); ++i) {
        if (i) samples_file << ',';
        samples_file << tensor.shape()[i];
    }
    samples_file << "],\"stride\":[";
    for (std::size_t i = 0; i < tensor.stride().size(); ++i) {
        if (i) samples_file << ',';
        samples_file << tensor.stride()[i];
    }
    samples_file << "],\"offset\":" << offset << ",\"bits\":[";
    for (std::size_t i = 0; i < host.numel(); ++i) {
        if (i) samples_file << ',';
        std::uint32_t bits = 0;
        std::memcpy(&bits, values + i, sizeof(T));
        samples_file << bits;
    }
    samples_file << "]}";
}

template <typename T>
void Export(const Tensor& result, const Tensor& a, const Tensor* b,
            const Precision& precision, const std::string& name,
            const Tensor* upstream = nullptr, int a_offset = 0, int b_offset = 0,
            int random_start = 0) {
    if (result.dtype() != precision.dtype || result.shape() != a.shape() ||
        result.device() != a.device())
        throw std::runtime_error(name + ": wrong output metadata");
    samples_file << "{\"name\":\"" << name << "\",\"dtype\":\""
                 << dtype_name(precision.dtype) << "\",\"device\":\""
                 << (a.device() == Device::kCPU ? "cpu" : "cuda") << "\",\"a\":";
    WriteTensor<T>(a, a_offset);
    if (b) { samples_file << ",\"b\":"; WriteTensor<T>(*b, b_offset); }
    if (upstream) { samples_file << ",\"g\":"; WriteTensor<T>(*upstream, 1); }
    samples_file << ",\"actual\":";
    WriteTensor<T>(result);
    // 只标记分组，不拆开 Tensor，保持计算顺序和 Torch 向量化路径不变。
    if (name.find("/special") != std::string::npos) {
        samples_file << ",\"cohorts\":{\"special\":" << result.numel() << '}';
    } else {
        samples_file << ",\"cohorts\":{\"basic\":513,\"boundary\":" << random_start - 513
                     << ",\"random\":" << result.numel() - random_start << '}';
    }
    samples_file << "}\n";
    exported += static_cast<int>(result.numel());
}

template <typename T>
void CheckUnary(Device device, const Precision& precision, const std::string& op) {
    const double epsilon = std::ldexp(1., -precision.fraction_bits);
    const double tiny = std::ldexp(1., precision.min_exponent - precision.fraction_bits);
    const double normal = std::ldexp(1., precision.min_exponent);
    const double largest = std::ldexp(2. - epsilon, precision.max_exponent);
    std::vector<double> samples = {-90, -20, -17, -12, -1, -0.001, 0, 0.001,
                                  1, 8, 12, 17, 20, 89, 90};
    if (op == "sqrt" || op == "rsqrt") samples = {1e-6, 0.001, 0.1, 1, 2, 11, 17, 300, 60000};
    // 补充 exp 在三种 dtype 上的上溢/下溢和舍入边界附近输入。
    if (op == "exp") {
        for (double v : {-104., -103., -93., -92., -17.34, -17.32,
                         11.08, 11.09, 88.5, 88.7, 88.75}) samples.push_back(v);
    }
    std::mt19937 rng(20261120);
    while (samples.size() < 513) {
        const double u = static_cast<double>(rng() % 100001) / 100000;
        samples.push_back(op == "sqrt" || op == "rsqrt" ? std::pow(10., -4 + 8*u) : -20 + 40*u);
    }
    std::vector<double> gradients;
    for (std::size_t i = 0; i < samples.size(); ++i)
        gradients.push_back(i % 3 == 0 ? 1. : i % 3 == 1 ? -0.5 : 3.);
    const std::vector<double> gradient_edges = {
        tiny, -tiny, normal, -normal, 0.5, -0.5, 3., -3., largest, -largest};
    std::vector<double> input_edges;
    if (op == "sqrt" || op == "rsqrt" || op == "neg") {
        input_edges = {tiny, 2*tiny, normal-tiny, normal, normal+tiny,
                       1-epsilon/2, 1., 1+epsilon, largest/2, largest};
        // 每一个二进制数量级都实际取样，不仅依赖随机覆盖。
        for (int e = precision.min_exponent - precision.fraction_bits;
             e <= precision.max_exponent; ++e)
            input_edges.push_back(std::ldexp(1.5, e));
        if (op == "neg") {
            const auto positive = input_edges;
            for (double v : positive) input_edges.push_back(-v);
        }
    } else {
        input_edges = {-200., -104., -90., -20., -1.2784645428, -1.,
                       -epsilon, 0., epsilon, 1., 12., 20., 89., 104., 200.};
        // exp 的最大值、最小正规数、最小次正规数及舍入为零的边界两侧。
        for (double boundary : {std::log(largest), std::log(normal),
                                std::log(tiny), std::log(tiny/2)}) {
            for (double delta : {-2*epsilon*std::abs(boundary), 0.,
                                  2*epsilon*std::abs(boundary)}) {
                input_edges.push_back(boundary + delta);
                input_edges.push_back(-boundary + delta);
            }
        }
        // SiLU 导数过零点附近，暴露相消误差。
        for (int i = -16; i <= 16; ++i)
            input_edges.push_back(-1.2784645428 + i*epsilon);
    }
    for (double v : input_edges) {
        for (double g : gradient_edges) {
            samples.push_back(v);
            gradients.push_back(g);
        }
    }
    const int random_start = static_cast<int>(samples.size());
    for (int i = 0; i < 4097; ++i) {
        const int e = precision.min_exponent - precision.fraction_bits +
                      rng() % (precision.max_exponent - precision.min_exponent +
                               precision.fraction_bits + 1);
        const double magnitude = std::min(largest, std::ldexp(1. + (rng()%65536)/65536., e));
        gradients.push_back(rng()%2 ? magnitude : -magnitude);
        if (op == "sqrt" || op == "rsqrt" || op == "neg") {
            const int xe = precision.min_exponent - precision.fraction_bits +
                           rng() % (precision.max_exponent - precision.min_exponent +
                                    precision.fraction_bits + 1);
            const double x = std::min(largest, std::ldexp(1. + (rng()%65536)/65536., xe));
            samples.push_back(op == "neg" && rng()%2 ? -x : x);
        } else {
            samples.push_back(-200. + 400.*(rng()%1000001)/1000000.);
        }
    }
    std::vector<T> values, upstream;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        values.emplace_back(static_cast<float>(samples[i]));
        upstream.emplace_back(static_cast<float>(gradients[i]));
    }
    const int count = static_cast<int>(values.size());
    // 用非连续输入和非连续上游梯度。
    Tensor storage(std::vector<int>{count, 2}, precision.dtype, device);
    Tensor x = storage.transpose(0, 1)[0];
    x.copy_(Tensor(values.data(), {count}, precision.dtype, device));
    x.requires_grad_(true);
    Tensor y;
    if (op == "exp") y = exp(x);
    else if (op == "sigmoid") y = sigmoid(x);
    else if (op == "silu") y = silu(x);
    else if (op == "sqrt") y = sqrt(x);
    else if (op == "neg") y = -x;
    else y = rsqrt(x);
    Tensor grad_storage(std::vector<int>{count, 2}, precision.dtype, device);
    Tensor grad = grad_storage.transpose(0, 1)[1];
    grad.copy_(Tensor(upstream.data(), {count}, precision.dtype, device));
    y.backward(grad);
    Export<T>(y, x, nullptr, precision, op+"/forward", nullptr, 0, 0, random_start);
    Export<T>(x.grad(), x, nullptr, precision, op+"/backward", &grad, 0, 0, random_start);
}

template <typename T>
void CheckBinary(Device device, const Precision& precision, const std::string& op) {
    const double epsilon = std::ldexp(1., -precision.fraction_bits);
    const double tiny = std::ldexp(1., precision.min_exponent - precision.fraction_bits);
    const double normal = std::ldexp(1., precision.min_exponent);
    const double largest = std::ldexp(2. - epsilon, precision.max_exponent);
    // 1/300 的分母梯度、平方下溢、大数加小数、相消、舍入边界。
    std::vector<double> av = {1, 1e-6, 1, 60000, 2048, 256, 1, 1, -1, 0};
    std::vector<double> bv = {300, 1e-4, 60000, 60000, 1, 1, 0.00048828125, -0.99951171875, 1, 1};
    std::mt19937 rng(20261121);
    while (av.size() < 513) {
        const double a = (static_cast<int>(rng() % 20001) - 10000) / 10000.;
        const double magnitude = std::pow(10., -4 + 8 * (rng() % 100001) / 100000.);
        av.push_back(a);
        bv.push_back(rng() % 2 ? magnitude : -magnitude);
    }
    std::vector<double> gv;
    for (std::size_t i = 0; i < av.size(); ++i)
        gv.push_back(i % 3 == 0 ? 1. : i % 3 == 1 ? -0.5 : 3.);
    auto append = [&](double a, double b, double g) {
        av.push_back(a); bv.push_back(b); gv.push_back(g);
    };
    // 两个操作数都取极端值；上游梯度独立变化，不再固定为三个小数。
    const std::vector<double> edges = {
        tiny, 2*tiny, normal-tiny, normal, normal+tiny, epsilon/2,
        0.5, 1-epsilon/2, 1., 1+epsilon, 2., 300.,
        std::sqrt(largest), largest/2, largest};
    const std::vector<double> gradient_edges = {
        tiny, -tiny, normal, -normal, 0.5, -0.5, 3., -3., largest, -largest};
    for (double a : edges) {
        for (double b : edges) {
            for (double g : gradient_edges) {
                for (int sa : {-1, 1}) {
                    for (int sb : {-1, 1}) append(sa*a, sb*b, g);
                }
            }
        }
    }
    // 相消、半 ULP 的两侧、预期上溢/下溢，以及零上游梯度。
    for (int e = precision.min_exponent - precision.fraction_bits;
         e <= precision.max_exponent; ++e) {
        const double value = std::ldexp(1., e);
        const double half_ulp = std::max(tiny, value*epsilon/2);
        for (double sign : {-1., 1.}) {
            append(sign*value, sign*value, largest);
            append(sign*value, -sign*value, tiny);
            append(sign*value, sign*value*(1+epsilon), -largest);
            for (double factor : {1-epsilon, 1., 1+epsilon}) {
                append(sign*value, half_ulp*factor, 3.);
                append(half_ulp*factor, sign*value, -0.5);
            }
            append(sign*value, tiny, 0.);
            append(sign*value, largest, -0.0);
            append(0., sign*value, largest);
        }
    }
    // 固定讨论过的反例，避免后续换顺序时只修好某一组输入。
    append(60000., 2., 3.);
    append(largest/2, tiny, tiny);
    append(tiny, largest/2, largest/2);
    if (precision.dtype != Dtype::kFloat16) {
        append(1e30, 1e-10, 1e-30);
        append(1e-30, 1e10, 1e30);
    }
    // a、b、g 各自按指数均匀采样，覆盖整个可表示指数范围。
    const int random_start = static_cast<int>(av.size());
    for (int i = 0; i < 4097; ++i) {
        double sample[3];
        for (double& v : sample) {
            const int e = precision.min_exponent - precision.fraction_bits +
                          rng() % (precision.max_exponent - precision.min_exponent +
                                   precision.fraction_bits + 1);
            v = std::min(largest, std::ldexp(1. + (rng()%65536)/65536., e));
            if (rng()%2) v = -v;
        }
        append(sample[0], sample[1], sample[2]);
    }
    std::vector<T> a, b, g;
    for (std::size_t i = 0; i < av.size(); ++i) {
        a.emplace_back(static_cast<float>(av[i])); b.emplace_back(static_cast<float>(bv[i]));
        g.emplace_back(static_cast<float>(gv[i]));
        if (!std::isfinite(static_cast<float>(a.back())) ||
            !std::isfinite(static_cast<float>(b.back())) ||
            !std::isfinite(static_cast<float>(g.back())) || static_cast<float>(b.back()) == 0)
            throw std::runtime_error("binary precision samples must be finite with nonzero b");
    }
    const int count = static_cast<int>(a.size());
    Tensor x_storage(std::vector<int>{count, 2}, precision.dtype, device);
    Tensor z_storage(std::vector<int>{count, 3}, precision.dtype, device);
    Tensor x = x_storage.transpose(0, 1)[1], z = z_storage.transpose(0, 1)[2];
    x.copy_(Tensor(a.data(), {count}, precision.dtype, device));
    z.copy_(Tensor(b.data(), {count}, precision.dtype, device));
    x.requires_grad_(true); z.requires_grad_(true);
    Tensor y;
    if (op == "add") y = x+z;
    else if (op == "sub") y = x-z;
    else if (op == "mul") y = x*z;
    else y = x/z;
    Tensor grad_storage(std::vector<int>{count, 2}, precision.dtype, device);
    Tensor grad = grad_storage.transpose(0, 1)[1];
    grad.copy_(Tensor(g.data(), {count}, precision.dtype, device));
    y.backward(grad);
    Export<T>(y, x, &z, precision, op+"/forward", nullptr, 1, 2, random_start);
    Export<T>(x.grad(), x, &z, precision, op+"/grad_a", &grad, 1, 2, random_start);
    Export<T>(z.grad(), x, &z, precision, op+"/grad_b", &grad, 1, 2, random_start);
}

template <typename T>
void CheckSpecialValues(Device device, const Precision& precision) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<T> values = {T(0.0f), T(-0.0f), T(inf), T(-inf), T(nan)};
    const Tensor x(values.data(), {5}, precision.dtype, device);
    Export<T>(exp(x), x, nullptr, precision, "exp/special");
    Export<T>(sigmoid(x), x, nullptr, precision, "sigmoid/special");
    Export<T>(silu(x), x, nullptr, precision, "silu/special");
    Export<T>(sqrt(x), x, nullptr, precision, "sqrt/special");
    Export<T>(rsqrt(x), x, nullptr, precision, "rsqrt/special");
    Export<T>(-x, x, nullptr, precision, "neg/special");

    // 除零及 NaN/Inf 的前向单独测，不在这些点执行反向。
    const std::vector<double> special = {-inf, -1., -0., 0., 1., inf, nan};
    std::vector<T> a, b;
    for (double v : special) {
        for (double w : special) {
            a.emplace_back(static_cast<float>(v));
            b.emplace_back(static_cast<float>(w));
        }
    }
    const int count = static_cast<int>(a.size());
    const Tensor left(a.data(), {count}, precision.dtype, device);
    const Tensor right(b.data(), {count}, precision.dtype, device);
    for (const std::string op : {"add", "sub", "mul", "div"}) {
        Tensor out;
        if (op == "add") out = left + right;
        else if (op == "sub") out = left - right;
        else if (op == "mul") out = left * right;
        else out = left / right;
        Export<T>(out, left, &right, precision, op+"/special");
    }
}

template <typename T>
void Run(Device device, Precision precision) {
    CheckSpecialValues<T>(device, precision);
    for (const std::string op : {"exp", "sigmoid", "silu", "sqrt", "rsqrt", "neg"})
        CheckUnary<T>(device, precision, op);
    for (const std::string op : {"add", "sub", "mul", "div"})
        CheckBinary<T>(device, precision, op);
}
}  // namespace

int main(int argc, char** argv) {
    using namespace mtensor;
    if (argc != 2) {
        std::cerr << "PyTorch comparison unavailable. Install tests/precision/requirements-torch.txt, "
                     "configure Python3_EXECUTABLE, then run CTest. "
                     "This executable only exports samples: <executable> <output.jsonl>\n";
        return 77;
    }
    try {
#ifdef PRECISION_TEST_CUDA
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
            (status == cudaSuccess && count == 0)) return 77;
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        constexpr auto device = Device::kCUDA;
#else
        constexpr auto device = Device::kCPU;
#endif
        samples_file.open(argv[1], std::ios::out | std::ios::trunc);
        if (!samples_file) throw std::runtime_error("cannot open sample output file");
        samples_file.exceptions(std::ios::badbit | std::ios::failbit);
        Run<float>(device, {Dtype::kFloat32, 23, -126, 127});
        Run<Float16>(device, {Dtype::kFloat16, 10, -14, 15});
        Run<BFloat16>(device, {Dtype::kBFloat16, 7, -126, 127});
        samples_file.close();
        std::cout << "Exported " << exported << " values for PyTorch comparison\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
