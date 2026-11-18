#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef MATMUL_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "ops/matmul.h"

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
    const Tensor result = lhs.matmul(rhs).to(Device::kCPU);
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
    const Tensor result = lhs.matmul(rhs).to(Device::kCPU);
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
