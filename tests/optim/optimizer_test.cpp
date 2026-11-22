#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef OPTIMIZER_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/optim/sgd.h"
#include "mini_infer/optim/adam.h"
#include "mini_infer/autograd.h"
#include "mini_infer/modules/linear.h"

namespace {
using namespace mtensor;
#ifdef OPTIMIZER_TEST_CUDA
constexpr Device kDevice = Device::kCUDA;
#else
constexpr Device kDevice = Device::kCPU;
#endif

void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<float> Values(const Tensor& tensor) {
    NogradNode guard;
    const Tensor host = tensor.contiguous().to(Device::kCPU);
    const auto* data = static_cast<const float*>(host.data_ptr());
    return {data, data + host.numel()};
}

void CheckValues(const Tensor& tensor, const std::vector<float>& expected) {
    const auto actual = Values(tensor);
    Require(actual.size() == expected.size(), "optimizer output element count");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        Require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= 2e-6f,
                "optimizer value differs from expected");
    }
}

void SetGradient(Tensor& parameter, const std::vector<float>& values) {
    NogradNode guard;
    parameter.zero_grad();
    parameter.accumulate_grad(Tensor(values.data(), parameter.shape(), Dtype::kFloat32, kDevice));
}

std::unique_ptr<Optimizer> MakeOptimizer(bool adam, std::vector<Tensor> parameters) {
    if (adam) return std::make_unique<Adam>(std::move(parameters), 0.1f, 0.5f, 0.5f, 0.0f);
    return std::make_unique<SGD>(std::move(parameters), 0.1f, 0.5f);
}

void CheckSGD() {
    const float initial[] = {1, -2};
    Tensor parameter(initial, {2}, Dtype::kFloat32, kDevice);
    parameter.requires_grad_(true);
    Tensor alias = parameter;
    SGD optimizer({parameter}, 0.1f, 0.5f);
    const void* parameter_data = parameter.data_ptr();
    const auto version = parameter.version();
    optimizer.zero_grad();
    optimizer.step();
    Require(!parameter.has_grad() && parameter.version() == version, "missing gradient changed parameter");

    SetGradient(parameter, {0.5f, -1.0f});
    optimizer.step();
    CheckValues(alias, {0.95f, -1.9f});
    CheckValues(parameter.grad(), {0.5f, -1.0f});
    Require(parameter.data_ptr() == parameter_data && parameter.version() == version + 1,
            "SGD replaced storage or missed version update");
    Require(parameter.requires_grad() && !parameter.grad_node() && !parameter.grad().grad_node(),
            "SGD changed autograd metadata or built a graph");
    const void* grad_data = parameter.grad().data_ptr();
    optimizer.zero_grad();
    Require(parameter.has_grad() && parameter.grad().data_ptr() == grad_data, "zero_grad replaced gradient storage");
    CheckValues(parameter.grad(), {0, 0});
    optimizer.step();
    CheckValues(parameter, {0.925f, -1.85f});
    SetGradient(parameter, {-0.25f, 0.5f});
    optimizer.step();
    CheckValues(parameter, {0.9375f, -1.875f});
    Require(EnableGrad(), "SGD did not restore grad mode");
    {
        NogradNode guard;
        optimizer.step();
        Require(!EnableGrad(), "SGD changed outer no-grad mode");
    }

    Tensor plain(initial, {2}, Dtype::kFloat32, kDevice);
    SGD without_momentum({plain}, 0.1f);
    SetGradient(plain, {0.5f, -1.0f});
    without_momentum.step();
    without_momentum.zero_grad();
    without_momentum.step();
    CheckValues(plain, {0.95f, -1.9f});
}

void CheckAdam() {
    Tensor first = full({}, 1.0f, Dtype::kFloat32, kDevice);
    Tensor late = full({}, 1.0f, Dtype::kFloat32, kDevice);
    Adam optimizer({first, late}, 0.1f, 0.5f, 0.5f, 0.0f);
    SetGradient(first, {2});
    optimizer.step();
    CheckValues(first, {0.9f});
    CheckValues(late, {1});
    Require(!late.has_grad(), "Adam allocated gradient for skipped parameter");

    optimizer.zero_grad();
    SetGradient(late, {2});
    optimizer.step();
    CheckValues(first, {0.9f - 0.1f / std::sqrt(3.0f)});
    CheckValues(late, {0.9f});
    Require(EnableGrad() && !first.grad_node() && !first.grad().grad_node(),
            "Adam built a graph or changed grad mode");
    {
        NogradNode guard;
        optimizer.step();
        Require(!EnableGrad(), "Adam changed outer no-grad mode");
    }

    Tensor zero = full({2}, 1.0f, Dtype::kFloat32, kDevice);
    Adam default_options({zero});
    SetGradient(zero, {0, 0});
    default_options.step();
    CheckValues(zero, {1, 1});
}

void CheckValidation() {
    auto rejects = [](auto action) {
        bool rejected = false;
        try { action(); } catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "optimizer accepted invalid arguments");
    };
    Tensor parameter = full({2}, 1.0f, Dtype::kFloat32, kDevice);
    rejects([&] { SGD optimizer({}, 0.1f); });
    rejects([&] { Adam optimizer({Tensor()}); });
    rejects([&] { SGD optimizer({parameter, parameter}, 0.1f); });
    rejects([&] { Adam optimizer({parameter, parameter.detach()}); });
    for (Dtype dtype : {Dtype::kFloat16, Dtype::kBFloat16, Dtype::kInt32}) {
        Tensor low = full({2}, 1, dtype, kDevice);
        rejects([&] { SGD optimizer({low}, 0.1f); });
        rejects([&] { Adam optimizer({low}); });
    }
    const Tensor overlapping = full({1}, 1.0f, Dtype::kFloat32, kDevice).broadcast({2});
    rejects([&] { SGD optimizer({overlapping}, 0.1f); });
    for (float bad : {-1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
        rejects([&] { SGD optimizer({parameter}, bad); });
        rejects([&] { SGD optimizer({parameter}, 0.1f, bad); });
        rejects([&] { Adam optimizer({parameter}, bad); });
        rejects([&] { Adam optimizer({parameter}, 0.1f, bad); });
        rejects([&] { Adam optimizer({parameter}, 0.1f, 0.9f, bad); });
        rejects([&] { Adam optimizer({parameter}, 0.1f, 0.9f, 0.999f, bad); });
    }
    rejects([&] { Adam optimizer({parameter}, 0.1f, 1.0f); });
    rejects([&] { Adam optimizer({parameter}, 0.1f, 0.9f, 1.0f); });

    SetGradient(parameter, {1, -1});
    SGD zero_sgd({parameter}, 0.0f, 0.9f);
    Adam zero_adam({parameter}, 0.0f);
    zero_sgd.step();
    zero_adam.step();
    CheckValues(parameter, {1, 1});

    for (bool adam : {false, true}) {
        std::vector<Tensor> invalid_gradients{
            full({3}, 1, Dtype::kFloat32, kDevice),
            full({2}, 1, Dtype::kFloat16, kDevice)};
#ifdef OPTIMIZER_TEST_CUDA
        invalid_gradients.push_back(full({2}, 1, Dtype::kFloat32, Device::kCPU));
#endif
        for (const auto& invalid : invalid_gradients) {
            Tensor first = full({2}, 1.0f, Dtype::kFloat32, kDevice);
            Tensor second = full({2}, 2.0f, Dtype::kFloat32, kDevice);
            auto optimizer = MakeOptimizer(adam, {first, second});
            SetGradient(first, {1, 1});
            {
                NogradNode guard;
                second.accumulate_grad(invalid);
            }
            rejects([&] { optimizer->step(); });
            CheckValues(first, {1, 1});
            CheckValues(second, {2, 2});
            Require(EnableGrad(), "gradient validation did not restore grad mode");
        }

        Tensor weight = full({2}, 1.0f, Dtype::kFloat32, kDevice);
        weight.requires_grad_(true);
        auto optimizer = MakeOptimizer(adam, {weight});
        Tensor loss = (weight * weight).sum();
        loss.backward();
        optimizer->step();
        bool rejected = false;
        try { loss.backward(); } catch (const std::logic_error&) { rejected = true; }
        Require(rejected, "optimizer update did not invalidate saved input versions");
        {
            NogradNode guard;
            optimizer->zero_grad();
            Require(!EnableGrad(), "zero_grad changed outer no-grad mode");
        }
    }
}

void CheckTraining(bool adam) {
    const float xv[] = {-1, -1, -1, 1, 1, -1, 1, 1};
    const float yv[] = {1.5f, -4.5f, 5.5f, -0.5f};
    Tensor x(xv, {4, 2}, Dtype::kFloat32, kDevice);
    Tensor target(yv, {4, 1}, Dtype::kFloat32, kDevice);
    Tensor weight = full({1, 2}, 0, Dtype::kFloat32, kDevice);
    Tensor bias = full({1}, 0, Dtype::kFloat32, kDevice);
    weight.requires_grad_(true);
    bias.requires_grad_(true);
    Linear model(weight, bias);
    std::unique_ptr<Optimizer> optimizer;
    if (adam) optimizer = std::make_unique<Adam>(std::vector<Tensor>{weight, bias}, 0.1f);
    else optimizer = std::make_unique<SGD>(std::vector<Tensor>{weight, bias}, 0.1f);
    const int steps = adam ? 250 : 80;
    for (int step = 0; step < steps; ++step) {
        optimizer->zero_grad();
        Tensor error = model(x) - target;
        Tensor loss = (error * error).mean();
        loss.backward();
        optimizer->step();
    }
    NogradNode guard;
    Tensor error = model(x) - target;
    const float final_loss = Values((error * error).mean())[0];
    const auto weights = Values(weight);
    const float intercept = Values(bias)[0];
    Require(final_loss < 1e-6f && std::abs(weights[0] - 2) < 2e-3f &&
            std::abs(weights[1] + 3) < 2e-3f && std::abs(intercept - 0.5f) < 2e-3f,
            "linear regression did not converge");
    std::cout << (adam ? "Adam" : "SGD") << " training: loss 13.25 -> " << final_loss
              << ", weight=[" << weights[0] << ", " << weights[1] << "], bias=" << intercept << '\n';
}

void WriteTensor(std::ostream& out, const Tensor& tensor, int offset = 0) {
    const auto values = Values(tensor);
    out << "{\"shape\":[";
    for (std::size_t i = 0; i < tensor.shape().size(); ++i) {
        if (i) out << ',';
        out << tensor.shape()[i];
    }
    out << "],\"stride\":[";
    for (std::size_t i = 0; i < tensor.stride().size(); ++i) {
        if (i) out << ',';
        out << tensor.stride()[i];
    }
    out << "],\"offset\":" << offset << ",\"bits\":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        std::uint32_t bits;
        std::memcpy(&bits, &values[i], sizeof(bits));
        out << bits;
    }
    out << "]}";
}

void ExportTraces(std::ostream& out) {
    struct Case {
        const char* name;
        bool adam;
        float lr, momentum, beta1, beta2, eps;
    };
    NogradNode guard;
    for (const Case& c : {Case{"sgd", false, .1f, 0, 0, 0, 0},
                          Case{"sgd_momentum", false, .05f, .9f, 0, 0, 0},
                          Case{"adam", true, .01f, 0, .9f, .999f, 1e-8f},
                          Case{"adam_custom", true, .03f, 0, .5f, .8f, 1e-5f},
                          Case{"adam_beta_zero", true, .02f, 0, 0, 0, 1e-8f}}) {
        Tensor backing = full({2, 3}, 7, Dtype::kFloat32, kDevice);
        Tensor parameter = backing.narrow(1, 1, 2);
        const float initial[] = {1, -2, .5f, 3};
        parameter.copy_(Tensor(initial, {2, 2}, Dtype::kFloat32, kDevice));
        Tensor late = full({}, -.75f, Dtype::kFloat32, kDevice);
        parameter.requires_grad_(true);
        late.requires_grad_(true);
        std::unique_ptr<Optimizer> optimizer;
        if (c.adam) optimizer = std::make_unique<Adam>(
            std::vector<Tensor>{parameter, late}, c.lr, c.beta1, c.beta2, c.eps);
        else optimizer = std::make_unique<SGD>(std::vector<Tensor>{parameter, late}, c.lr, c.momentum);
        out << "{\"name\":\"" << c.name << "\",\"device\":\""
            << (kDevice == Device::kCPU ? "cpu" : "cuda") << "\",\"lr\":" << c.lr
            << ",\"momentum\":" << c.momentum << ",\"beta1\":" << c.beta1
            << ",\"beta2\":" << c.beta2 << ",\"eps\":" << c.eps << ",\"initial\":[";
        WriteTensor(out, parameter, 1);
        out << ',';
        WriteTensor(out, late);
        out << "],\"steps\":[";
        for (int step = 0; step < 10; ++step) {
            optimizer->zero_grad();
            std::vector<float> gradient(4);
            for (int j = 0; j < 4; ++j) {
                gradient[j] = step == 4 ? 0.0f : ((step + j) % 7 - 3) * 0.125f;
                if (step == 7) gradient[j] *= 16;
            }
            SetGradient(parameter, gradient);
            if (step >= 3) SetGradient(late, {(step % 3 - 1) * 0.25f});
            if (step) out << ',';
            out << "{\"gradients\":[";
            WriteTensor(out, parameter.grad());
            out << ',';
            if (late.has_grad()) WriteTensor(out, late.grad());
            else out << "null";
            optimizer->step();
            out << "],\"parameters\":[";
            WriteTensor(out, parameter, 1);
            out << ',';
            WriteTensor(out, late);
            out << "]}";
            const auto storage = Values(backing);
            Require(storage[0] == 7 && storage[3] == 7, "optimizer wrote outside strided parameter");
        }
        out << "]}\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
#ifdef OPTIMIZER_TEST_CUDA
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
            (status == cudaSuccess && count == 0)) return 77;
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
        if (argc == 1) {
            CheckSGD();
            CheckAdam();
            CheckValidation();
            CheckTraining(false);
            CheckTraining(true);
        } else if (argc == 2) {
            std::ofstream out(argv[1]);
            if (!out) throw std::runtime_error("cannot open optimizer trace output");
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out << std::setprecision(9);
            ExportTraces(out);
        } else {
            throw std::invalid_argument("usage: optimizer_test [traces.jsonl]");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
