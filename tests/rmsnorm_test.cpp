#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef RMSNORM_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/modules/rmsnorm.h"
#include "mini_infer/autograd.h"
#include "mini_infer/ops/norm.h"

namespace {
using namespace mtensor;
#ifdef RMSNORM_TEST_CUDA
constexpr Device kDevice = Device::kCUDA;
#else
constexpr Device kDevice = Device::kCPU;
#endif

void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void CheckStructure() {
    const float xv[] = {1, -1, 1, -1};
    const float wv[] = {2, -1};
    const float gv[] = {1, 2, 3, 4};
    Tensor x(xv, {2, 2}, Dtype::kFloat32, kDevice);
    Tensor w(wv, {2}, Dtype::kFloat32, kDevice);
    x.requires_grad_(true); w.requires_grad_(true);
    RMSNorm layer(w, 3.0f);
    Module& module = layer;
    Tensor y = module(x);
    Require(y.shape() == x.shape() && y.dtype() == x.dtype() && y.device() == kDevice,
            "RMSNorm output metadata");
    y.backward(Tensor(gv, {2, 2}, Dtype::kFloat32, kDevice));
    const float expected_y[] = {1, .5f, 1, .5f};
    const float expected_x[] = {.75f, -.75f, 2.375f, -1.375f};
    const float expected_w[] = {2, -3};
    const Tensor tensors[] = {y, x.grad(), w.grad()};
    const float* expected[] = {expected_y, expected_x, expected_w};
    for (int k = 0; k < 3; ++k) {
        const Tensor host = tensors[k].contiguous().to(Device::kCPU);
        const auto* data = static_cast<const float*>(host.data_ptr());
        for (std::size_t i = 0; i < host.numel(); ++i)
            Require(data[i] == expected[k][i], "RMSNorm forward/input/weight gradient");
    }
    // Module 保存共享权重句柄；同一个图再次反传应累加到原始参数。
    y.backward(Tensor(gv, {2, 2}, Dtype::kFloat32, kDevice));
    const Tensor accumulated = w.grad().to(Device::kCPU);
    const auto* dw = static_cast<const float*>(accumulated.data_ptr());
    Require(dw[0] == 4 && dw[1] == -6, "RMSNorm gradient accumulation");
    {
        NogradNode guard;
        Require(!module(x).grad_node() && !rms_norm(x, w, 3).grad_node(),
                "RMSNorm built a graph under no-grad");
    }
    for (bool change_weight : {false, true}) {
        Tensor output = module(x);
        if (change_weight) w.fill_(1);
        else x.fill_(1);
        bool rejected = false;
        try { output.backward(Tensor(gv, {2, 2}, Dtype::kFloat32, kDevice)); }
        catch (const std::logic_error&) { rejected = true; }
        Require(rejected, "RMSNorm accepted an input/weight modified after forward");
    }
    auto rejects = [](auto action) {
        try { action(); } catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("RMSNorm accepted invalid shape");
    };
    Tensor matrix_weight({1, 2}, Dtype::kFloat32, kDevice);
    rejects([&] { RMSNorm invalid(matrix_weight, 1e-5f); });
    rejects([&] { rms_norm(x, matrix_weight, 1e-5f); });
    Tensor wrong_length({3}, Dtype::kFloat32, kDevice);
    rejects([&] { RMSNorm invalid(wrong_length, 1e-5f); invalid(x); });
    Tensor scalar(xv, {}, Dtype::kFloat32, kDevice);
    rejects([&] { module(scalar); });
}

template <typename T>
void WriteTensor(std::ostream& out, const Tensor& tensor, int offset = 0) {
    const Tensor host = tensor.contiguous().to(Device::kCPU);
    const auto* values = static_cast<const T*>(host.data_ptr());
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
    for (std::size_t i = 0; i < host.numel(); ++i) {
        if (i) out << ',';
        std::uint32_t bits = 0;
        std::memcpy(&bits, values + i, sizeof(T));
        out << bits;
    }
    out << "]}";
}

struct Case {
    const char* name;
    std::vector<int> shape;
    float eps = 1e-5f;
    float scale = 1;
    bool strided = false;
};

template <typename T>
void ExportCases(std::ostream& out, Dtype dtype) {
    for (const Case& c : {Case{"vector", {7}}, Case{"matrix", {3, 8}},
                          Case{"tensor3d", {2, 3, 7}},
                          Case{"strided", {2, 3, 7}, 1e-5f, 1, true},
                          Case{"single_feature", {3, 1}, .1f},
                          Case{"zeros", {2, 7}, 1e-5f, 0},
                          Case{"large_eps", {3, 8}, .25f},
                          Case{"small_input", {2, 8}, 1e-5f, 1e-4f},
                          Case{"large_input", {2, 8}, 1e-5f, 300},
                          Case{"wide", {2, 257}}}) {
        std::size_t count = 1;
        for (int d : c.shape) count *= d;
        const int h = c.shape.back();
        std::vector<T> xv, wv, gv;
        for (std::size_t i = 0; i < count; ++i) {
            xv.emplace_back(c.scale * (static_cast<int>(i % 13) - 6) / 4.0f);
            gv.emplace_back((static_cast<int>(i % 7) - 3) / 4.0f);
        }
        for (int j = 0; j < h; ++j) wv.emplace_back((j % 5 - 2) / 2.0f);
        Tensor x(xv.data(), c.shape, dtype, kDevice);
        Tensor w(wv.data(), {h}, dtype, kDevice);
        Tensor g(gv.data(), c.shape, dtype, kDevice);
        if (c.strided) {
            auto shape = c.shape;
            shape.push_back(2);
            Tensor xs(shape, dtype, kDevice);
            x = xs.narrow(static_cast<int>(shape.size()) - 1, 1, 1)
                  .squeeze(static_cast<int>(shape.size()) - 1);
            shape.back() = 3;
            Tensor gs(shape, dtype, kDevice);
            g = gs.narrow(static_cast<int>(shape.size()) - 1, 2, 1)
                  .squeeze(static_cast<int>(shape.size()) - 1);
            Tensor ws(std::vector<int>{h, 2}, dtype, kDevice);
            w = ws.transpose(0, 1)[1];
            x.copy_(Tensor(xv.data(), c.shape, dtype, kDevice));
            w.copy_(Tensor(wv.data(), {h}, dtype, kDevice));
            g.copy_(Tensor(gv.data(), c.shape, dtype, kDevice));
        }
        x.requires_grad_(true); w.requires_grad_(true);
        RMSNorm layer(w, c.eps);
        Tensor y = layer(x);
        const Tensor direct = rms_norm(x, w, c.eps);
        Require(y.shape() == x.shape() && y.dtype() == dtype && y.device() == kDevice,
                "RMSNorm numerical output metadata");
        const Tensor yh = y.contiguous().to(Device::kCPU);
        const Tensor dh = direct.contiguous().to(Device::kCPU);
        Require(std::memcmp(yh.data_ptr(), dh.data_ptr(), yh.nbytes()) == 0,
                "RMSNorm module differs from functional operator");
        y.backward(g);
        Require(x.grad().shape() == x.shape() && w.grad().shape() == w.shape() &&
                x.grad().dtype() == dtype && w.grad().dtype() == dtype &&
                x.grad().device() == kDevice && w.grad().device() == kDevice,
                "RMSNorm numerical gradient metadata");
        out << "{\"case\":\"" << c.name << "\",\"dtype\":\"" << dtype_name(dtype)
            << "\",\"device\":\"" << (kDevice == Device::kCPU ? "cpu" : "cuda")
            << "\",\"eps\":" << c.eps << ",\"x\":";
        WriteTensor<T>(out, x, c.strided ? 1 : 0);
        out << ",\"weight\":"; WriteTensor<T>(out, w, c.strided ? 1 : 0);
        out << ",\"g\":"; WriteTensor<T>(out, g, c.strided ? 2 : 0);
        out << ",\"forward\":"; WriteTensor<T>(out, y);
        out << ",\"grad_x\":"; WriteTensor<T>(out, x.grad());
        out << ",\"grad_weight\":"; WriteTensor<T>(out, w.grad());
        out << "}\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
#ifdef RMSNORM_TEST_CUDA
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
            (status == cudaSuccess && count == 0)) return 77;
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
        if (argc == 1) {
            CheckStructure();
        } else if (argc == 2) {
            std::ofstream out(argv[1]);
            if (!out) throw std::runtime_error("cannot open RMSNorm sample output");
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out.precision(9);  // Preserve the FP32 eps value in the JSON reference.
            ExportCases<float>(out, Dtype::kFloat32);
            ExportCases<Float16>(out, Dtype::kFloat16);
            ExportCases<BFloat16>(out, Dtype::kBFloat16);
        } else {
            throw std::invalid_argument("usage: rmsnorm_test [samples.jsonl]");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
