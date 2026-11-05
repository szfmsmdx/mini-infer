#include <cmath>
#include <cstddef>
#include <cstdint>
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

template <typename T>
void CheckValues(const Tensor& tensor, const std::vector<T>& expected) {
    Check(tensor.numel() == expected.size(), "Unexpected element count");
    const T* values = static_cast<const T*>(tensor.data_ptr());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        Check(values[i] == expected[i], "Unexpected element value");
    }
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

template <typename T>
void TestNativeBinary(
    BinaryOp operation,
    Dtype dtype,
    const std::vector<T>& left,
    const std::vector<T>& right,
    const std::vector<T>& expected) {
    const std::vector<int> shape = {static_cast<int>(left.size())};
    Tensor a(left.data(), shape, dtype);
    Tensor b(right.data(), shape, dtype);
    Tensor result = operation(a, b);

    Check(result.shape() == shape, "binary op must preserve shape");
    Check(result.dtype() == dtype, "binary op must preserve dtype");
    Check(result.device() == Device::kCPU, "binary op must preserve device");
    Check(!result.empty(), "binary op output must own storage");
    Check(result.data_ptr() != a.data_ptr() && result.data_ptr() != b.data_ptr(),
          "binary op output must use independent storage");

    CheckValues<T>(result, expected);
    CheckValues<T>(a, left);
    CheckValues<T>(b, right);

    result.fill_(0);
    CheckValues<T>(a, left);
    CheckValues<T>(b, right);
}

template <typename T>
void TestNativeAdd(
    Dtype dtype,
    const std::vector<T>& left,
    const std::vector<T>& right,
    const std::vector<T>& expected) {
    TestNativeBinary(mtensor::add, dtype, left, right, expected);
}

template <typename T>
void TestNativeSub(
    Dtype dtype,
    const std::vector<T>& left,
    const std::vector<T>& right,
    const std::vector<T>& expected) {
    TestNativeBinary(mtensor::sub, dtype, left, right, expected);
}

template <typename T>
void TestNativeMul(
    Dtype dtype,
    const std::vector<T>& left,
    const std::vector<T>& right,
    const std::vector<T>& expected) {
    TestNativeBinary(mtensor::mul, dtype, left, right, expected);
}

template <typename T>
void TestNativeDiv(
    Dtype dtype,
    const std::vector<T>& left,
    const std::vector<T>& right,
    const std::vector<T>& expected) {
    TestNativeBinary(mtensor::div, dtype, left, right, expected);
}

void TestFloat32Add() {
    TestNativeAdd<float>(
        Dtype::kFloat32,
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {5.0f, 7.0f, 9.0f});

    float left[] = {1, 2, 3, 4, 5, 6};
    float right[] = {10, 20, 30, 40, 50, 60};
    Tensor a(left, {2, 3}, Dtype::kFloat32);
    Tensor b(right, {2, 3}, Dtype::kFloat32);
    Tensor result = mtensor::add(a, b);
    Check(result.shape() == std::vector<int>({2, 3}), "add must preserve 2D shape");
    CheckValues<float>(result, {11, 22, 33, 44, 55, 66});

    float x = 1.5f;
    float y = 2.25f;
    Tensor scalar_a(&x, {}, Dtype::kFloat32);
    Tensor scalar_b(&y, {}, Dtype::kFloat32);
    Tensor scalar_result = scalar_a + scalar_b;
    Check(scalar_result.dim() == 0 && scalar_result.numel() == 1 &&
              !scalar_result.empty(),
          "Scalar add must keep a zero-dimensional result");
    CheckValues<float>(scalar_result, {3.75f});

    float matrix_data[] = {1, 2, 3, 4, 5, 6};
    Tensor matrix(matrix_data, {2, 3}, Dtype::kFloat32);
    Tensor view_result = matrix[0] + matrix[1];
    CheckValues<float>(view_result, {5, 7, 9});
    CheckValues<float>(matrix, {1, 2, 3, 4, 5, 6});
}

void TestFloat32Sub() {
    TestNativeSub<float>(
        Dtype::kFloat32,
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {-3.0f, -3.0f, -3.0f});

    float left[] = {1, 2, 3, 4, 5, 6};
    float right[] = {10, 20, 30, 40, 50, 60};
    Tensor a(left, {2, 3}, Dtype::kFloat32);
    Tensor b(right, {2, 3}, Dtype::kFloat32);
    Tensor result = mtensor::sub(a, b);
    Check(result.shape() == std::vector<int>({2, 3}), "sub must preserve 2D shape");
    CheckValues<float>(result, {-9, -18, -27, -36, -45, -54});

    float x = 1.5f;
    float y = 2.25f;
    Tensor scalar_a(&x, {}, Dtype::kFloat32);
    Tensor scalar_b(&y, {}, Dtype::kFloat32);
    Tensor scalar_result = scalar_a - scalar_b;
    Check(scalar_result.dim() == 0 && scalar_result.numel() == 1 &&
              !scalar_result.empty(),
          "Scalar sub must keep a zero-dimensional result");
    CheckValues<float>(scalar_result, {-0.75f});

    float matrix_data[] = {1, 2, 3, 4, 5, 6};
    Tensor matrix(matrix_data, {2, 3}, Dtype::kFloat32);
    Tensor view_result = matrix[1] - matrix[0];
    CheckValues<float>(view_result, {3, 3, 3});
    CheckValues<float>(matrix, {1, 2, 3, 4, 5, 6});
}

void TestFloat32Mul() {
    TestNativeMul<float>(
        Dtype::kFloat32,
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {4.0f, 10.0f, 18.0f});

    float left[] = {1, 2, 3, 4, 5, 6};
    float right[] = {10, 20, 30, 40, 50, 60};
    Tensor a(left, {2, 3}, Dtype::kFloat32);
    Tensor b(right, {2, 3}, Dtype::kFloat32);
    Tensor result = mtensor::mul(a, b);
    Check(result.shape() == std::vector<int>({2, 3}), "mul must preserve 2D shape");
    CheckValues<float>(result, {10, 40, 90, 160, 250, 360});

    float x = 1.5f;
    float y = 2.25f;
    Tensor scalar_a(&x, {}, Dtype::kFloat32);
    Tensor scalar_b(&y, {}, Dtype::kFloat32);
    Tensor scalar_result = scalar_a * scalar_b;
    Check(scalar_result.dim() == 0 && scalar_result.numel() == 1 &&
              !scalar_result.empty(),
          "Scalar mul must keep a zero-dimensional result");
    CheckValues<float>(scalar_result, {3.375f});

    float matrix_data[] = {1, 2, 3, 4, 5, 6};
    Tensor matrix(matrix_data, {2, 3}, Dtype::kFloat32);
    Tensor view_result = matrix[0] * matrix[1];
    CheckValues<float>(view_result, {4, 10, 18});
    CheckValues<float>(matrix, {1, 2, 3, 4, 5, 6});
}

void TestFloat32Div() {
    TestNativeDiv<float>(
        Dtype::kFloat32,
        {4.0f, 9.0f, 16.0f},
        {2.0f, 3.0f, 4.0f},
        {2.0f, 3.0f, 4.0f});

    float left[] = {1, 2, 3, 4, 5, 6};
    float right[] = {2, 2, 2, 2, 2, 2};
    Tensor a(left, {2, 3}, Dtype::kFloat32);
    Tensor b(right, {2, 3}, Dtype::kFloat32);
    Tensor result = mtensor::div(a, b);
    Check(result.shape() == std::vector<int>({2, 3}), "div must preserve 2D shape");
    CheckValues<float>(result, {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f});

    float x = 7.5f;
    float y = 2.5f;
    Tensor scalar_a(&x, {}, Dtype::kFloat32);
    Tensor scalar_b(&y, {}, Dtype::kFloat32);
    Tensor scalar_result = scalar_a / scalar_b;
    Check(scalar_result.dim() == 0 && scalar_result.numel() == 1 &&
              !scalar_result.empty(),
          "Scalar div must keep a zero-dimensional result");
    CheckValues<float>(scalar_result, {3.0f});

    float matrix_data[] = {1, 2, 3, 4, 5, 6};
    Tensor matrix(matrix_data, {2, 3}, Dtype::kFloat32);
    Tensor view_result = matrix[1] / matrix[0];
    CheckValues<float>(view_result, {4, 2.5f, 2});
    CheckValues<float>(matrix, {1, 2, 3, 4, 5, 6});
}

void TestIntegerAdd() {
    TestNativeAdd<std::int8_t>(Dtype::kInt8, {-3, 7}, {10, -2}, {7, 5});
    TestNativeAdd<std::uint8_t>(Dtype::kUInt8, {1, 250}, {2, 5}, {3, 255});
    TestNativeAdd<std::int16_t>(
        Dtype::kInt16,
        {-1000, 2000},
        {500, -3000},
        {-500, -1000});
    TestNativeAdd<std::int32_t>(
        Dtype::kInt32,
        {-100000, 200000},
        {50000, -300000},
        {-50000, -100000});
    TestNativeAdd<std::int64_t>(
        Dtype::kInt64,
        {std::int64_t{4000000000}, std::int64_t{-5000000000}},
        {std::int64_t{1000000000}, std::int64_t{2000000000}},
        {std::int64_t{5000000000}, std::int64_t{-3000000000}});
}

void TestIntegerWrap() {
    TestNativeAdd<std::int8_t>(Dtype::kInt8, {127, -128}, {1, -1}, {-128, 127});
    TestNativeAdd<std::uint8_t>(Dtype::kUInt8, {255, 0}, {1, 255}, {0, 255});

    const std::int16_t int16_max = std::numeric_limits<std::int16_t>::max();
    const std::int16_t int16_min = std::numeric_limits<std::int16_t>::min();
    TestNativeAdd<std::int16_t>(
        Dtype::kInt16,
        {int16_max, int16_min},
        {1, -1},
        {int16_min, int16_max});

    const std::int32_t int32_max = std::numeric_limits<std::int32_t>::max();
    const std::int32_t int32_min = std::numeric_limits<std::int32_t>::min();
    TestNativeAdd<std::int32_t>(
        Dtype::kInt32,
        {int32_max, int32_min},
        {1, -1},
        {int32_min, int32_max});

    const std::int64_t int64_max = std::numeric_limits<std::int64_t>::max();
    const std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    TestNativeAdd<std::int64_t>(
        Dtype::kInt64,
        {int64_max, int64_min},
        {1, -1},
        {int64_min, int64_max});
}

void TestIntegerSub() {
    TestNativeSub<std::int8_t>(Dtype::kInt8, {-3, 7}, {10, -2}, {-13, 9});
    TestNativeSub<std::uint8_t>(Dtype::kUInt8, {1, 250}, {2, 5}, {255, 245});
    TestNativeSub<std::int16_t>(
        Dtype::kInt16,
        {-1000, 2000},
        {500, -3000},
        {-1500, 5000});
    TestNativeSub<std::int32_t>(
        Dtype::kInt32,
        {-100000, 200000},
        {50000, -300000},
        {-150000, 500000});
    TestNativeSub<std::int64_t>(
        Dtype::kInt64,
        {std::int64_t{4000000000}, std::int64_t{-5000000000}},
        {std::int64_t{1000000000}, std::int64_t{2000000000}},
        {std::int64_t{3000000000}, std::int64_t{-7000000000}});
}

void TestIntegerSubWrap() {
    TestNativeSub<std::int8_t>(Dtype::kInt8, {-128, 127}, {1, -1}, {127, -128});
    TestNativeSub<std::uint8_t>(Dtype::kUInt8, {0, 255}, {1, 255}, {255, 0});

    const std::int16_t int16_max = std::numeric_limits<std::int16_t>::max();
    const std::int16_t int16_min = std::numeric_limits<std::int16_t>::min();
    TestNativeSub<std::int16_t>(
        Dtype::kInt16,
        {int16_min, int16_max},
        {1, -1},
        {int16_max, int16_min});

    const std::int32_t int32_max = std::numeric_limits<std::int32_t>::max();
    const std::int32_t int32_min = std::numeric_limits<std::int32_t>::min();
    TestNativeSub<std::int32_t>(
        Dtype::kInt32,
        {int32_min, int32_max},
        {1, -1},
        {int32_max, int32_min});

    const std::int64_t int64_max = std::numeric_limits<std::int64_t>::max();
    const std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    TestNativeSub<std::int64_t>(
        Dtype::kInt64,
        {int64_min, int64_max},
        {1, -1},
        {int64_max, int64_min});
}

void TestIntegerMul() {
    TestNativeMul<std::int8_t>(Dtype::kInt8, {3, -4}, {5, 6}, {15, -24});
    TestNativeMul<std::uint8_t>(Dtype::kUInt8, {3, 10}, {5, 20}, {15, 200});
    TestNativeMul<std::int16_t>(
        Dtype::kInt16,
        {-7, 100},
        {9, -20},
        {-63, -2000});
    TestNativeMul<std::int32_t>(
        Dtype::kInt32,
        {-1000, 2000},
        {3, -4},
        {-3000, -8000});
    TestNativeMul<std::int64_t>(
        Dtype::kInt64,
        {std::int64_t{1000000}, std::int64_t{-2000000}},
        {3, 4},
        {std::int64_t{3000000}, std::int64_t{-8000000}});
}

void TestIntegerMulWrap() {
    TestNativeMul<std::int8_t>(
        Dtype::kInt8,
        {127, -128, 100},
        {2, -1, 100},
        {-2, -128, 16});
    TestNativeMul<std::uint8_t>(
        Dtype::kUInt8,
        {255, 200},
        {2, 2},
        {254, 144});

    const std::int16_t int16_max = std::numeric_limits<std::int16_t>::max();
    const std::int16_t int16_min = std::numeric_limits<std::int16_t>::min();
    TestNativeMul<std::int16_t>(
        Dtype::kInt16,
        {int16_max, int16_min, int16_max},
        {2, 2, int16_max},
        {-2, 0, 1});

    const std::int32_t int32_max = std::numeric_limits<std::int32_t>::max();
    const std::int32_t int32_min = std::numeric_limits<std::int32_t>::min();
    TestNativeMul<std::int32_t>(
        Dtype::kInt32,
        {int32_max, int32_min},
        {2, 2},
        {-2, 0});

    const std::int64_t int64_max = std::numeric_limits<std::int64_t>::max();
    const std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    TestNativeMul<std::int64_t>(
        Dtype::kInt64,
        {int64_max, int64_min},
        {2, 2},
        {-2, 0});
}

void TestIntegerDiv() {
    TestNativeDiv<std::int8_t>(
        Dtype::kInt8,
        {7, -7, 7, -7},
        {2, 2, -2, -2},
        {3, -3, -3, 3});
    TestNativeDiv<std::uint8_t>(Dtype::kUInt8, {255, 10}, {2, 3}, {127, 3});
    TestNativeDiv<std::int16_t>(
        Dtype::kInt16,
        {-1000, 2000},
        {3, -4},
        {-333, -500});
    TestNativeDiv<std::int32_t>(
        Dtype::kInt32,
        {-100000, 200000},
        {3, -4},
        {-33333, -50000});
    TestNativeDiv<std::int64_t>(
        Dtype::kInt64,
        {std::int64_t{-5000000000}, std::int64_t{4000000000}},
        {3, -8},
        {std::int64_t{-1666666666}, std::int64_t{-500000000}});
}

void TestIntegerDivEdge() {
    TestNativeDiv<std::int8_t>(
        Dtype::kInt8,
        {7, -7, -128},
        {0, 0, -1},
        {0, 0, -128});
    TestNativeDiv<std::uint8_t>(
        Dtype::kUInt8,
        {255, 0},
        {0, 0},
        {0, 0});

    const std::int16_t int16_min = std::numeric_limits<std::int16_t>::min();
    TestNativeDiv<std::int16_t>(
        Dtype::kInt16,
        {int16_min, 7},
        {-1, 0},
        {int16_min, 0});

    const std::int32_t int32_min = std::numeric_limits<std::int32_t>::min();
    TestNativeDiv<std::int32_t>(
        Dtype::kInt32,
        {int32_min, -7},
        {-1, 0},
        {int32_min, 0});

    const std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    TestNativeDiv<std::int64_t>(
        Dtype::kInt64,
        {int64_min, -7},
        {-1, 0},
        {int64_min, 0});
}

void TestFloatSpecialValues() {
    const float float_max = std::numeric_limits<float>::max();
    Tensor max_a(&float_max, {}, Dtype::kFloat32);
    Tensor max_b(&float_max, {}, Dtype::kFloat32);
    Tensor overflow = max_a + max_b;
    const float* overflow_value = static_cast<const float*>(overflow.data_ptr());
    Check(std::isinf(*overflow_value) && *overflow_value > 0,
          "Float32 overflow must produce positive infinity");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float one = 1.0f;
    Tensor nan_a(&nan, {}, Dtype::kFloat32);
    Tensor one_b(&one, {}, Dtype::kFloat32);
    Tensor nan_result = nan_a + one_b;
    Check(std::isnan(*static_cast<const float*>(nan_result.data_ptr())),
          "Float32 NaN must propagate");

    const float negative_float_max = -float_max;
    Tensor negative_max_a(&negative_float_max, {}, Dtype::kFloat32);
    Tensor negative_overflow = negative_max_a - max_b;
    const float* negative_value =
        static_cast<const float*>(negative_overflow.data_ptr());
    Check(std::isinf(*negative_value) && *negative_value < 0,
          "Float32 negative overflow must produce negative infinity");

    Tensor nan_sub = nan_a - one_b;
    Check(std::isnan(*static_cast<const float*>(nan_sub.data_ptr())),
          "Float32 NaN must propagate through sub");

    Tensor mul_overflow = max_a * max_b;
    Check(std::isinf(*static_cast<const float*>(mul_overflow.data_ptr())),
          "Float32 multiplication overflow must produce infinity");

    const float zero = 0.0f;
    const float negative_one = -1.0f;
    Tensor zero_b(&zero, {}, Dtype::kFloat32);
    Tensor negative_one_b(&negative_one, {}, Dtype::kFloat32);
    Tensor div_inf = one_b / zero_b;
    Check(std::isinf(*static_cast<const float*>(div_inf.data_ptr())) &&
              *static_cast<const float*>(div_inf.data_ptr()) > 0,
          "Float32 division by zero must produce positive infinity");
    Tensor div_negative_inf = negative_one_b / zero_b;
    Check(std::isinf(*static_cast<const float*>(div_negative_inf.data_ptr())) &&
              *static_cast<const float*>(div_negative_inf.data_ptr()) < 0,
          "Float32 negative division by zero must produce negative infinity");

    Tensor zero_div_zero = zero_b / zero_b;
    Check(std::isnan(*static_cast<const float*>(zero_div_zero.data_ptr())),
          "Float32 zero divided by zero must produce NaN");
}

void CheckHalfBinary(
    BinaryOp operation,
    Dtype dtype,
    const std::vector<std::uint16_t>& left,
    const std::vector<std::uint16_t>& right,
    const std::vector<std::uint16_t>& expected) {
    const std::vector<int> shape = {static_cast<int>(left.size())};
    Tensor a(left.data(), shape, dtype);
    Tensor b(right.data(), shape, dtype);
    Tensor result = operation(a, b);

    Check(result.shape() == shape, "half binary op must preserve shape");
    Check(result.dtype() == dtype, "half binary op must preserve dtype");
    CheckValues<std::uint16_t>(a, left);
    CheckValues<std::uint16_t>(b, right);

    const auto* actual = static_cast<const std::uint16_t*>(result.data_ptr());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        Check(actual[i] == expected[i], "Unexpected half binary-op encoding");
    }
}

void CheckHalfAdd(
    Dtype dtype,
    const std::vector<std::uint16_t>& left,
    const std::vector<std::uint16_t>& right,
    const std::vector<std::uint16_t>& expected) {
    CheckHalfBinary(mtensor::add, dtype, left, right, expected);
}

void CheckHalfSub(
    Dtype dtype,
    const std::vector<std::uint16_t>& left,
    const std::vector<std::uint16_t>& right,
    const std::vector<std::uint16_t>& expected) {
    CheckHalfBinary(mtensor::sub, dtype, left, right, expected);
}

void CheckHalfMul(
    Dtype dtype,
    const std::vector<std::uint16_t>& left,
    const std::vector<std::uint16_t>& right,
    const std::vector<std::uint16_t>& expected) {
    CheckHalfBinary(mtensor::mul, dtype, left, right, expected);
}

void CheckHalfDiv(
    Dtype dtype,
    const std::vector<std::uint16_t>& left,
    const std::vector<std::uint16_t>& right,
    const std::vector<std::uint16_t>& expected) {
    CheckHalfBinary(mtensor::div, dtype, left, right, expected);
}

void TestHalfAdd() {
    // 1.0 + 2.0 = 3.0，1.5 + 2.25 = 3.75。
    CheckHalfAdd(
        Dtype::kFloat16,
        {0x3c00u, 0x3e00u},
        {0x4000u, 0x4080u},
        {0x4200u, 0x4380u});
    CheckHalfAdd(
        Dtype::kBFloat16,
        {0x3f80u, 0x3fc0u},
        {0x4000u, 0x4010u},
        {0x4040u, 0x4070u});

    // 最大有限值相加后按所在格式舍入为无穷。
    CheckHalfAdd(Dtype::kFloat16, {0x7bffu}, {0x7bffu}, {0x7c00u});
    CheckHalfAdd(Dtype::kBFloat16, {0x7f7fu}, {0x7f7fu}, {0x7f80u});
}

void TestHalfSub() {
    // 3.0 - 1.0 = 2.0，1.0 - 2.0 = -1.0。
    CheckHalfSub(
        Dtype::kFloat16,
        {0x4200u, 0x3c00u},
        {0x3c00u, 0x4000u},
        {0x4000u, 0xbc00u});
    CheckHalfSub(
        Dtype::kBFloat16,
        {0x4040u, 0x3f80u},
        {0x3f80u, 0x4000u},
        {0x4000u, 0xbf80u});

    // 负数溢出按所在格式舍入为负无穷。
    CheckHalfSub(Dtype::kFloat16, {0xfbffu}, {0x7bffu}, {0xfc00u});
    CheckHalfSub(Dtype::kBFloat16, {0xff7fu}, {0x7f7fu}, {0xff80u});
}

void TestHalfMul() {
    // 1.0 * 2.0 = 2.0，2.0 * 3.0 = 6.0。
    CheckHalfMul(Dtype::kFloat16, {0x3c00u}, {0x4000u}, {0x4000u});
    CheckHalfMul(Dtype::kFloat16, {0x4000u}, {0x4200u}, {0x4600u});
    CheckHalfMul(Dtype::kBFloat16, {0x3f80u}, {0x4000u}, {0x4000u});
    CheckHalfMul(Dtype::kBFloat16, {0x4000u}, {0x4040u}, {0x40c0u});

    // 最大有限值相乘后按所在格式舍入为无穷。
    CheckHalfMul(Dtype::kFloat16, {0x7bffu}, {0x7bffu}, {0x7c00u});
    CheckHalfMul(Dtype::kBFloat16, {0x7f7fu}, {0x7f7fu}, {0x7f80u});
}

void CheckHalfDivNaN(Dtype dtype) {
    const std::uint16_t zero = 0x0000u;
    Tensor a(&zero, {}, dtype);
    Tensor result = mtensor::div(a, a);
    const std::uint16_t bits =
        *static_cast<const std::uint16_t*>(result.data_ptr());
    const float value = dtype == Dtype::kFloat16
        ? mtensor::DecodeFloat16(&bits)
        : mtensor::DecodeBFloat16(&bits);
    Check(std::isnan(value), "half zero divided by zero must produce NaN");
}

void TestHalfDiv() {
    // 6.0 / 2.0 = 3.0，1.0 / 2.0 = 0.5。
    CheckHalfDiv(Dtype::kFloat16, {0x4600u}, {0x4000u}, {0x4200u});
    CheckHalfDiv(Dtype::kFloat16, {0x3c00u}, {0x4000u}, {0x3800u});
    CheckHalfDiv(Dtype::kBFloat16, {0x40c0u}, {0x4000u}, {0x4040u});
    CheckHalfDiv(Dtype::kBFloat16, {0x3f80u}, {0x4000u}, {0x3f00u});

    // 非零数除以零得到无穷；零除以零得到 NaN。
    CheckHalfDiv(Dtype::kFloat16, {0x3c00u}, {0x0000u}, {0x7c00u});
    CheckHalfDiv(Dtype::kBFloat16, {0x3f80u}, {0x0000u}, {0x7f80u});
    CheckHalfDivNaN(Dtype::kFloat16);
    CheckHalfDivNaN(Dtype::kBFloat16);
}

void TestNeg() {
    float input[] = {1.0f, -2.0f, 0.0f};
    Tensor a(input, {3}, Dtype::kFloat32);
    Tensor result = -a;

    Check(result.shape() == a.shape(), "neg must preserve shape");
    Check(result.dtype() == a.dtype(), "neg must preserve dtype");
    Check(result.device() == a.device(), "neg must preserve device");
    Check(!result.empty(), "neg output must own storage");
    Check(result.data_ptr() != a.data_ptr(), "neg output must use independent storage");
    CheckValues<float>(result, {-1.0f, 2.0f, 0.0f});
    CheckValues<float>(a, {1.0f, -2.0f, 0.0f});

    result.fill_(0);
    CheckValues<float>(a, {1.0f, -2.0f, 0.0f});

    float scalar_value = 3.5f;
    Tensor scalar(&scalar_value, {}, Dtype::kFloat32);
    Tensor scalar_result = -scalar;
    Check(scalar_result.dim() == 0 && scalar_result.numel() == 1 &&
              !scalar_result.empty(),
          "Scalar neg must keep a zero-dimensional result");
    CheckValues<float>(scalar_result, {-3.5f});

    std::int8_t int8_input[] = {-128, 127, 0};
    Tensor int8_tensor(int8_input, {3}, Dtype::kInt8);
    Tensor int8_result = -int8_tensor;
    CheckValues<std::int8_t>(int8_result, {-128, -127, 0});

    std::uint8_t uint8_input[] = {0, 1, 255};
    Tensor uint8_tensor(uint8_input, {3}, Dtype::kUInt8);
    Tensor uint8_result = -uint8_tensor;
    CheckValues<std::uint8_t>(uint8_result, {0, 255, 1});

    const std::int32_t int32_min = std::numeric_limits<std::int32_t>::min();
    std::int32_t int32_input[] = {int32_min, 0};
    Tensor int32_tensor(int32_input, {2}, Dtype::kInt32);
    Tensor int32_result = -int32_tensor;
    CheckValues<std::int32_t>(int32_result, {int32_min, 0});

    const std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    std::int64_t int64_input[] = {int64_min, 0};
    Tensor int64_tensor(int64_input, {2}, Dtype::kInt64);
    Tensor int64_result = -int64_tensor;
    CheckValues<std::int64_t>(int64_result, {int64_min, 0});

    // FP16/BF16 的取负是翻转符号位，零、无穷也一并翻转。
    std::uint16_t float16_input[] = {0x3c00u, 0xbc00u, 0x0000u, 0x8000u, 0x7c00u};
    Tensor float16_tensor(float16_input, {5}, Dtype::kFloat16);
    Tensor float16_result = -float16_tensor;
    CheckValues<std::uint16_t>(
        float16_result, {0xbc00u, 0x3c00u, 0x8000u, 0x0000u, 0xfc00u});

    std::uint16_t bfloat16_input[] = {0x3f80u, 0xbf80u, 0x7f80u};
    Tensor bfloat16_tensor(bfloat16_input, {3}, Dtype::kBFloat16);
    Tensor bfloat16_result = -bfloat16_tensor;
    CheckValues<std::uint16_t>(bfloat16_result, {0xbf80u, 0x3f80u, 0xff80u});

    Tensor empty;
    bool threw = false;
    try {
        Tensor invalid = -empty;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "Unary minus on an empty Tensor must throw");
}

void TestErrors() {
    float input[] = {1, 2};
    Tensor a(input, {2}, Dtype::kFloat32);
    Tensor empty;

    float longer_input[] = {1, 2, 3};
    Tensor b(longer_input, {3}, Dtype::kFloat32);

    std::int32_t integer_input[] = {1, 2};
    Tensor integer(integer_input, {2}, Dtype::kInt32);

    const BinaryOp operations[] = {
        mtensor::add, mtensor::sub, mtensor::mul, mtensor::div};
    for (BinaryOp operation : operations) {
        CheckBinaryThrows(operation, empty, a);
        CheckBinaryThrows(operation, a, empty);
        CheckBinaryThrows(operation, a, b);
        CheckBinaryThrows(operation, a, integer);
    }

    CheckValues<float>(a, {1, 2});
    CheckValues<float>(b, {1, 2, 3});
    CheckValues<std::int32_t>(integer, {1, 2});
}

}  // namespace

int main() {
    try {
        TestFloat32Add();
        TestFloat32Sub();
        TestFloat32Mul();
        TestFloat32Div();
        TestIntegerAdd();
        TestIntegerSub();
        TestIntegerMul();
        TestIntegerDiv();
        TestIntegerWrap();
        TestIntegerSubWrap();
        TestIntegerMulWrap();
        TestIntegerDivEdge();
        TestFloatSpecialValues();
        TestHalfAdd();
        TestHalfSub();
        TestHalfMul();
        TestHalfDiv();
        TestNeg();
        TestErrors();
        std::cout << "Elementwise CPU tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
