#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ops/elementwise.h"
#include "tensor/tensor.h"

namespace {

using mtensor::Device;
using mtensor::Dtype;
using mtensor::Tensor;

void Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T>
std::vector<T> Read(Tensor& tensor) {
    std::vector<T> result(tensor.numel());
    if (!result.empty()) {
        CheckCuda(cudaMemcpy(result.data(), tensor.data_ptr(), tensor.nbytes(),
                             cudaMemcpyDeviceToHost));
    }
    return result;
}

template <typename T>
void CheckValues(Tensor& tensor, const std::vector<T>& expected) {
    Check(Read<T>(tensor) == expected, "CUDA Tensor elements differ from expected values");
}

using BinaryOp = Tensor (*)(const Tensor&, const Tensor&);

void CheckBinaryThrows(BinaryOp operation, const Tensor& a, const Tensor& b) {
    bool threw = false;
    try {
        operation(a, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "Incompatible operation must throw std::invalid_argument");
}

void CheckSameBits(const Tensor& expected, const Tensor& actual, const char* message) {
    Check(expected.nbytes() == actual.nbytes(), message);
    Check(std::memcmp(expected.data_ptr(), actual.data_ptr(), expected.nbytes()) == 0, message);
}

template <typename T>
void TestBinaryValues(Dtype dtype, const std::vector<int>& shape,
                      const std::vector<T>& left, const std::vector<T>& right) {
    Tensor a(left.data(), shape, dtype);
    Tensor b(right.data(), shape, dtype);
    Tensor ga = a.to(Device::kCUDA);
    Tensor gb = b.to(Device::kCUDA);

    for (BinaryOp op : {mtensor::add, mtensor::sub, mtensor::mul, mtensor::div}) {
        Tensor expected = op(a, b);
        Tensor actual = op(ga, gb).to(Device::kCPU);
        Check(actual.shape() == expected.shape(), "binary op must preserve shape");
        Check(actual.dtype() == dtype, "binary op must preserve dtype");
        CheckSameBits(expected, actual, "CUDA binary result differs from CPU");
    }

    Tensor result = mtensor::add(ga, gb);
    Check(result.device() == Device::kCUDA, "binary op output must stay on CUDA");
    Check(result.data_ptr() != ga.data_ptr() && result.data_ptr() != gb.data_ptr(),
          "binary op output must use independent storage");
    result.fill_(0);
    CheckValues<T>(ga, left);
    CheckValues<T>(gb, right);

    Tensor expected_neg = mtensor::neg(a);
    Tensor actual_neg = mtensor::neg(ga).to(Device::kCPU);
    CheckSameBits(expected_neg, actual_neg, "CUDA neg result differs from CPU");
}

template <typename T>
void TestStridedValues(Dtype dtype, const std::vector<T>& left, const std::vector<T>& right) {
    Tensor a(left.data(), {2, 3}, dtype);
    Tensor b(right.data(), {2, 2}, dtype);
    Tensor av = a.narrow(1, 1, 2);
    Tensor ga = a.to(Device::kCUDA);
    Tensor gb = b.to(Device::kCUDA);
    Tensor gv = ga.narrow(1, 1, 2);

    for (BinaryOp op : {mtensor::add, mtensor::sub, mtensor::mul, mtensor::div}) {
        Tensor expected = op(av, b);
        Tensor actual = op(gv, gb).to(Device::kCPU);
        CheckSameBits(expected, actual, "CUDA strided binary result differs from CPU");

        Tensor reverse_expected = op(b, av);
        Tensor reverse_actual = op(gb, gv).to(Device::kCPU);
        CheckSameBits(reverse_expected, reverse_actual,
                      "CUDA reverse strided binary result differs from CPU");
    }

    Tensor expected_neg = mtensor::neg(av);
    Tensor actual_neg = mtensor::neg(gv).to(Device::kCPU);
    CheckSameBits(expected_neg, actual_neg, "CUDA strided neg result differs from CPU");

    CheckValues<T>(ga, left);
    CheckValues<T>(gb, right);
}

std::uint16_t EncodeHalf(Dtype dtype, float value) {
    return dtype == Dtype::kFloat16 ? mtensor::EncodeFloat16(value)
                                    : mtensor::EncodeBFloat16(value);
}

float DecodeHalf(Dtype dtype, std::uint16_t bits) {
    return dtype == Dtype::kFloat16 ? mtensor::DecodeFloat16(&bits)
                                    : mtensor::DecodeBFloat16(&bits);
}

void TestFloat32() {
    TestBinaryValues<float>(
        Dtype::kFloat32,
        {3},
        {1.5f, -2.0f, 8.0f},
        {2.0f, 4.0f, -2.0f});
}

void TestIntegers() {
    TestBinaryValues<std::int8_t>(
        Dtype::kInt8, {2, 2},
        {127, -128, 100, -7},
        {1, -1, 100, 2});
    TestBinaryValues<std::uint8_t>(
        Dtype::kUInt8, {2, 2},
        {255, 0, 200, 7},
        {1, 255, 2, 3});
    TestBinaryValues<std::int16_t>(
        Dtype::kInt16, {2, 2},
        {32767, -32768, -1000, 2000},
        {1, -1, 3, -4});
    TestBinaryValues<std::int32_t>(
        Dtype::kInt32, {2, 2},
        {INT32_MAX, INT32_MIN, -100000, 200000},
        {1, -1, 3, -4});
    TestBinaryValues<std::int64_t>(
        Dtype::kInt64, {2, 2},
        {INT64_MAX, INT64_MIN, std::int64_t{-5000000000}, std::int64_t{4000000000}},
        {1, -1, 3, -8});
}

void TestHalf() {
    for (Dtype dtype : {Dtype::kFloat16, Dtype::kBFloat16}) {
        TestBinaryValues<std::uint16_t>(
            dtype,
            {3},
            {EncodeHalf(dtype, 0.0f), EncodeHalf(dtype, 1.5f), EncodeHalf(dtype, -2.25f)},
            {EncodeHalf(dtype, 2.0f), EncodeHalf(dtype, 4.0f), EncodeHalf(dtype, 2.0f)});

        const std::uint16_t max_bits =
            dtype == Dtype::kFloat16 ? 0x7bffu : 0x7f7fu;
        Tensor a(&max_bits, {}, dtype, Device::kCUDA);
        Tensor overflow = mtensor::add(a, a).to(Device::kCPU);
        const auto overflow_bits =
            *static_cast<const std::uint16_t*>(overflow.data_ptr());
        Check(std::isinf(DecodeHalf(dtype, overflow_bits)),
              "half overflow must produce infinity");

        const std::uint16_t zero = 0;
        Tensor z(&zero, {}, dtype, Device::kCUDA);
        Tensor nan_result = mtensor::div(z, z).to(Device::kCPU);
        const auto nan_bits = *static_cast<const std::uint16_t*>(nan_result.data_ptr());
        Check(std::isnan(DecodeHalf(dtype, nan_bits)),
              "half zero divided by zero must produce NaN");
    }
}

void TestFloatSpecialValues() {
    const float max = std::numeric_limits<float>::max();
    float value = max;
    Tensor a(&value, {}, Dtype::kFloat32, Device::kCUDA);
    Tensor overflow = mtensor::add(a, a).to(Device::kCPU);
    const float* overflow_value = static_cast<const float*>(overflow.data_ptr());
    Check(std::isinf(*overflow_value) && *overflow_value > 0,
          "Float32 overflow must produce positive infinity");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float one = 1.0f;
    float nan_value = nan;
    float one_value = one;
    Tensor nan_tensor(&nan_value, {}, Dtype::kFloat32, Device::kCUDA);
    Tensor one_tensor(&one_value, {}, Dtype::kFloat32, Device::kCUDA);
    Tensor nan_result = mtensor::add(nan_tensor, one_tensor).to(Device::kCPU);
    Check(std::isnan(*static_cast<const float*>(nan_result.data_ptr())),
          "Float32 NaN must propagate");

    const float zero = 0.0f;
    const float negative_one = -1.0f;
    float zero_value = zero;
    float negative_one_value = negative_one;
    Tensor zero_tensor(&zero_value, {}, Dtype::kFloat32, Device::kCUDA);
    Tensor negative_one_tensor(&negative_one_value, {}, Dtype::kFloat32, Device::kCUDA);

    Tensor div_inf = mtensor::div(one_tensor, zero_tensor).to(Device::kCPU);
    const float* div_inf_value = static_cast<const float*>(div_inf.data_ptr());
    Check(std::isinf(*div_inf_value) && *div_inf_value > 0,
          "Float32 division by zero must produce positive infinity");

    Tensor div_negative_inf =
        mtensor::div(negative_one_tensor, zero_tensor).to(Device::kCPU);
    const float* div_negative_inf_value =
        static_cast<const float*>(div_negative_inf.data_ptr());
    Check(std::isinf(*div_negative_inf_value) && *div_negative_inf_value < 0,
          "Float32 negative division by zero must produce negative infinity");

    Tensor div_nan = mtensor::div(zero_tensor, zero_tensor).to(Device::kCPU);
    Check(std::isnan(*static_cast<const float*>(div_nan.data_ptr())),
          "Float32 zero divided by zero must produce NaN");
}

void TestStrided() {
    TestStridedValues<float>(
        Dtype::kFloat32,
        {0, 1, 2, 3, 4, 5},
        {10, 20, 30, 40});
    TestStridedValues<std::int8_t>(
        Dtype::kInt8,
        {0, 127, -128, 3, -4, 5},
        {1, -1, 0, 2});
    TestStridedValues<std::uint8_t>(
        Dtype::kUInt8,
        {0, 255, 128, 3, 4, 5},
        {1, 255, 0, 2});
    TestStridedValues<std::int16_t>(
        Dtype::kInt16,
        {0, 32767, -32768, 3, -4, 5},
        {1, -1, 0, 2});
    TestStridedValues<std::int32_t>(
        Dtype::kInt32,
        {0, INT32_MAX, INT32_MIN, 3, -4, 5},
        {1, -1, 0, 2});
    TestStridedValues<std::int64_t>(
        Dtype::kInt64,
        {0, INT64_MAX, INT64_MIN, 3, -4, 5},
        {1, -1, 0, 2});
    for (Dtype dtype : {Dtype::kFloat16, Dtype::kBFloat16}) {
        TestStridedValues<std::uint16_t>(
            dtype,
            {EncodeHalf(dtype, 0.0f), EncodeHalf(dtype, 1.0f), EncodeHalf(dtype, 2.0f),
             EncodeHalf(dtype, 3.0f), EncodeHalf(dtype, 4.0f), EncodeHalf(dtype, 5.0f)},
            {EncodeHalf(dtype, 10.0f), EncodeHalf(dtype, 20.0f),
             EncodeHalf(dtype, 30.0f), EncodeHalf(dtype, 40.0f)});
    }
}

void TestScalar() {
    float x = 3.0f;
    float y = 2.0f;
    Tensor a(&x, {}, Dtype::kFloat32, Device::kCUDA);
    Tensor b(&y, {}, Dtype::kFloat32, Device::kCUDA);

    Tensor sum = mtensor::add(a, b);
    Check(sum.dim() == 0 && sum.numel() == 1 && !sum.empty(),
          "Scalar add must keep a zero-dimensional result");
    CheckValues<float>(sum, {5.0f});

    Tensor quotient = mtensor::div(a, b);
    Check(quotient.dim() == 0 && quotient.numel() == 1 && !quotient.empty(),
          "Scalar div must keep a zero-dimensional result");
    CheckValues<float>(quotient, {1.5f});
}

void TestBlockBoundary() {
    std::vector<float> input(2 * 514, 2.0f);
    Tensor wide(input.data(), {2, 514}, Dtype::kFloat32, Device::kCUDA);
    Tensor view = wide.narrow(1, 1, 513);
    Tensor sum = mtensor::add(view, view);
    CheckValues<float>(sum, std::vector<float>(1026, 4.0f));
}

void TestErrors() {
    float input[] = {1, 2};
    Tensor cuda(input, {2}, Dtype::kFloat32, Device::kCUDA);
    Tensor cpu(input, {2}, Dtype::kFloat32);
    Tensor empty;

    float longer_input[] = {1, 2, 3};
    Tensor cuda_longer(longer_input, {3}, Dtype::kFloat32, Device::kCUDA);

    std::int32_t integer_input[] = {1, 2};
    Tensor cuda_integer(integer_input, {2}, Dtype::kInt32, Device::kCUDA);

    const BinaryOp operations[] = {
        mtensor::add, mtensor::sub, mtensor::mul, mtensor::div};
    for (BinaryOp operation : operations) {
        CheckBinaryThrows(operation, empty, cuda);
        CheckBinaryThrows(operation, cuda, empty);
        CheckBinaryThrows(operation, cuda, cuda_longer);
        CheckBinaryThrows(operation, cuda, cuda_integer);
        CheckBinaryThrows(operation, cuda, cpu);
    }

    bool threw = false;
    try {
        mtensor::neg(empty);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "Unary minus on an empty Tensor must throw");
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && device_count == 0)) {
            std::cout << "Skipping CUDA elementwise tests: no usable GPU/driver\n";
            return 77;
        }
        CheckCuda(status);
        TestFloat32();
        TestIntegers();
        TestHalf();
        TestFloatSpecialValues();
        TestStrided();
        TestScalar();
        TestBlockBoundary();
        TestErrors();
        CheckCuda(cudaDeviceSynchronize());
        std::cout << "Elementwise CUDA tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
