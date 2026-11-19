#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef MATMUL_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/ops/matmul.h"
#include "mini_infer/autograd.h"

namespace {
using namespace mtensor;

#ifdef MATMUL_TEST_CUDA
constexpr Device kDevice = Device::kCUDA;
#else
constexpr Device kDevice = Device::kCPU;
#endif

template <typename T>
void CheckInteger(Dtype dtype) {
    const T a[] = {std::numeric_limits<T>::max(), 1};
    const T b[] = {2, 3};
    const Tensor lhs(a, {1, 2}, dtype, kDevice);
    const Tensor rhs(b, {2, 1}, dtype, kDevice);
    const Tensor out = lhs.matmul(rhs);
    if (out.grad_node()) throw std::runtime_error("integer matmul created a grad node");
    const Tensor result = out.to(Device::kCPU);
    if (result.dtype() != dtype || *static_cast<const T*>(result.data_ptr()) != 1) {
        throw std::runtime_error("integer matmul wrapping");
    }
}

template <auto Encode, auto Decode>
void CheckHalf(Dtype dtype) {
    const std::uint16_t a[] = {Encode(256), Encode(0.0625f), Encode(-256)};
    const std::uint16_t b[] = {Encode(1), Encode(1), Encode(1)};
    const Tensor lhs(a, {1, 3}, dtype, kDevice);
    const Tensor rhs(b, {3, 1}, dtype, kDevice);
    const Tensor out = lhs.matmul(rhs);
    if (!out.grad_node()) throw std::runtime_error("half matmul missing grad node");
    const Tensor result = out.to(Device::kCPU);
    if (result.dtype() != dtype || Decode(result.data_ptr()) != 0.0625f) {
        throw std::runtime_error("half matmul FP32 accumulation");
    }
}

void CheckFloat() {
    const float a[] = {1, 4, 2, 5, 3, 6};
    const float b[] = {7, 9, 11, 8, 10, 12};
    const Tensor lhs(a, {3, 2}, Dtype::kFloat32, kDevice);
    const Tensor rhs(b, {2, 3}, Dtype::kFloat32, kDevice);
    const Tensor result = matmul(lhs.transpose(0, 1), rhs.transpose(0, 1)).to(Device::kCPU);
    const auto* values = static_cast<const float*>(result.data_ptr());
    if (result.shape() != std::vector<int>{2, 2} || result.dtype() != Dtype::kFloat32 ||
        values[0] != 58 || values[1] != 64 || values[2] != 139 || values[3] != 154) {
        throw std::runtime_error("strided FP32 matmul");
    }
    try {
        lhs.matmul(rhs.transpose(0, 1));
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("matmul accepted mismatched inner dimensions");
}

template <typename T>
void CheckGradient(const char* name, const Tensor& input, const std::vector<float>& expected) {
    const std::string label = std::string(dtype_name(input.dtype())) + " " + name;
    if (!input.has_grad()) throw std::runtime_error(label + ": missing gradient");
    const Tensor grad = input.grad();
    if (grad.shape() != input.shape() || grad.dtype() != input.dtype() ||
        grad.device() != input.device() || grad.numel() != expected.size()) {
        throw std::runtime_error(label + ": incorrect gradient metadata");
    }
    const Tensor host = grad.to(Device::kCPU).contiguous();
    const auto* data = static_cast<const T*>(host.data_ptr());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float value = static_cast<float>(data[i]);
        if (value != expected[i]) {
            throw std::runtime_error(label + "[" + std::to_string(i) + "]: expected " +
                                     std::to_string(expected[i]) + ", got " + std::to_string(value));
        }
    }
}

void CheckBackward() {
    const float a[] = {1, 2, 3, 4, 5, 6};
    const float b[] = {1, 2, 0, 1, 2, -1};
    const float g[] = {2, 3, 5, 7};
    Tensor lhs(a, {2, 3}, Dtype::kFloat32, kDevice);
    Tensor rhs(b, {3, 2}, Dtype::kFloat32, kDevice);
    const Tensor upstream(g, {2, 2}, Dtype::kFloat32, kDevice);
    lhs.requires_grad_(true);
    rhs.requires_grad_(true);
    Tensor out = lhs.matmul(rhs);
    out.backward(upstream);

    CheckGradient<float>("dA", lhs, {8, 3, 1, 19, 7, 3});
    CheckGradient<float>("dB", rhs, {22, 31, 29, 41, 36, 51});

    for (Tensor* input : {&lhs, &rhs}) {
        out = lhs.matmul(rhs);
        input->transpose(0, 1).zero_();
        try {
            out.backward(upstream);
        } catch (const std::logic_error& err) {
            if (std::string(err.what()).find("modified after forward") == std::string::npos) throw;
            continue;
        }
        throw std::runtime_error("matmul backward accepted a modified input");
    }
    NogradNode guard;
    if (lhs.matmul(rhs).grad_node()) throw std::runtime_error("matmul ignored no-grad guard");
}

template <typename T>
void CheckLinearBackward(Dtype dtype) {
    const auto tensor = [dtype](const std::vector<int>& shape, std::initializer_list<float> values) {
        const std::vector<T> stored(values.begin(), values.end());
        return Tensor(stored.data(), shape, dtype, kDevice);
    };
    Tensor x = tensor({2, 3}, {1, -2, 0.5f, 3, 1, -1});
    Tensor w = tensor({4, 3}, {1, 0.5f, -1, -2, 1, 0, 0.5f, 3, 2, 1, -1, 0.5f});
    const Tensor g = tensor({2, 4}, {0.5f, -1, 2, 3, -2, 1, 0.5f, -1});
    x.requires_grad_(true);
    w.requires_grad_(true);
    x.matmul(w.transpose(0, 1)).backward(g);

    // dX = G @ W; dW = G.T @ X, in the original leaf's [4, 3] layout.
    // All reference values are exactly representable in both FP16 and BF16.
    CheckGradient<T>("dX", x, {6.5f, 2.25f, 5, -4.75f, 2.5f, 2.5f});
    CheckGradient<T>("dW", w, {-5.5f, -3, 2.25f, 2, 3, -1.5f, 3.5f, -3.5f, 0.5f, 0, -7, 2.5f});
}

void CheckLargerOutput() {
    constexpr int m = 17;
    constexpr int k = 5;
    constexpr int n = 19;
    const std::vector<float> a(m * k, 2.0f);
    const std::vector<float> b(k * n, 3.0f);
    const Tensor lhs(a.data(), {m, k}, Dtype::kFloat32, kDevice);
    const Tensor rhs(b.data(), {k, n}, Dtype::kFloat32, kDevice);
    const Tensor result = lhs.matmul(rhs).to(Device::kCPU);
    const auto* values = static_cast<const float*>(result.data_ptr());
    for (int i = 0; i < m * n; ++i) {
        if (values[i] != 30.0f) throw std::runtime_error("matmul result for 17 x 19 output");
    }
}
}  // namespace

int main() {
    try {
#ifdef MATMUL_TEST_CUDA
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
            (status == cudaSuccess && count == 0)) return 77;
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
        CheckFloat();
        CheckBackward();
        CheckLinearBackward<float>(Dtype::kFloat32);
        CheckLinearBackward<Float16>(Dtype::kFloat16);
        CheckLinearBackward<BFloat16>(Dtype::kBFloat16);
        CheckLargerOutput();
        CheckHalf<EncodeFloat16, DecodeFloat16>(Dtype::kFloat16);
        CheckHalf<EncodeBFloat16, DecodeBFloat16>(Dtype::kBFloat16);
        CheckInteger<std::int8_t>(Dtype::kInt8);
        CheckInteger<std::uint8_t>(Dtype::kUInt8);
        CheckInteger<std::int16_t>(Dtype::kInt16);
        CheckInteger<std::int32_t>(Dtype::kInt32);
        CheckInteger<std::int64_t>(Dtype::kInt64);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
