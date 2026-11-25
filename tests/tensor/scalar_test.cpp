#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "mini_infer/autograd.h"
#include "ops/reduction/nodes.h"
#include "mini_infer/tensor/scalar.h"
#include "mini_infer/tensor/tensor.h"

namespace {

void Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void TestScalar() {
    using mtensor::Scalar;
    static_assert(std::is_convertible_v<int, Scalar>);
    static_assert(std::is_convertible_v<unsigned long long, Scalar>);
    static_assert(std::is_convertible_v<float, Scalar>);
    static_assert(std::is_convertible_v<double, Scalar>);

    const std::int64_t large = (std::int64_t{1} << 60) + 3;
    Check(Scalar(large).to<std::int64_t>() == large, "signed scalar precision");
    Check(Scalar(std::numeric_limits<std::uint64_t>::max()).to<std::uint64_t>() ==
              std::numeric_limits<std::uint64_t>::max(), "unsigned scalar precision");
    Check(Scalar(-12).to<double>() == -12.0, "signed to floating point");
    Check(Scalar(1.5f).to<float>() == 1.5f, "floating-point scalar");
    Check(Scalar(-3.75).to<std::int32_t>() == -3, "floating point truncates toward zero");
    Check(Scalar(-0.5).to<std::uint8_t>() == 0, "negative fraction truncates to zero");

    for (double value : {128.0, -129.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        bool threw = false;
        try {
            (void)Scalar(value).to<std::int8_t>();
        } catch (const std::out_of_range&) {
            threw = true;
        }
        Check(threw, "invalid float-to-integer conversion must throw");
    }
}

void TestFill() {
    using namespace mtensor;
    Tensor integer({2}, Dtype::kInt64);
    const std::int64_t large = (std::int64_t{1} << 60) + 3;
    const Scalar value(large);
    Check(&integer.fill_(value) == &integer, "fill returns its Tensor");
    const auto* data = static_cast<const std::int64_t*>(integer.data_ptr());
    Check(data[0] == large && data[1] == large, "fill preserves int64 precision");

    Tensor small({2}, Dtype::kInt8);
    small.fill_(-3.75);
    const auto* small_data = static_cast<const std::int8_t*>(small.data_ptr());
    Check(small_data[0] == -3 && small_data[1] == -3, "fill converts floating-point input");
    bool threw = false;
    try {
        small.fill_(128.0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    Check(threw && small_data[0] == -3 && small_data[1] == -3,
          "invalid conversion leaves Tensor unchanged");

    Tensor scalar(std::vector<int>{}, Dtype::kFloat32);
    scalar.fill_(1.5);
    Check(scalar.dim() == 0 && *static_cast<const float*>(scalar.data_ptr()) == 1.5f,
          "fill zero-dimensional Tensor");
}

void TestScalarTensorConstruction() {
    using namespace mtensor;
    static_assert(!std::is_convertible_v<Scalar, Tensor>);
    Tensor floating(1.5f);
    Check(floating.shape().empty() && floating.numel() == 1 &&
          floating.device() == Device::kCPU && floating.dtype() == Dtype::kFloat32 &&
          *static_cast<const float*>(floating.data_ptr()) == 1.5f,
          "floating-point scalar Tensor construction");
    Tensor floating_double(2.5);
    Check(floating_double.dtype() == Dtype::kFloat32 &&
          *static_cast<const float*>(floating_double.data_ptr()) == 2.5f,
          "double uses default Float32");
    Tensor small_integer(42);
    Check(small_integer.dtype() == Dtype::kInt64 &&
          *static_cast<const std::int64_t*>(small_integer.data_ptr()) == 42,
          "int uses default Int64");
    const std::int64_t large = (std::int64_t{1} << 60) + 3;
    Tensor integer(large);
    Check(integer.shape().empty() && integer.dtype() == Dtype::kInt64 &&
          *static_cast<const std::int64_t*>(integer.data_ptr()) == large,
          "integer scalar Tensor construction preserves precision");
    Tensor unsigned_max(static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
    Check(*static_cast<const std::int64_t*>(unsigned_max.data_ptr()) ==
              std::numeric_limits<std::int64_t>::max(), "maximum unsigned Int64 value");
    bool threw = false;
    try {
        Tensor too_large(std::numeric_limits<std::uint64_t>::max());
    } catch (const std::out_of_range&) {
        threw = true;
    }
    Check(threw, "unsigned scalar outside Int64 range must throw");
    Tensor empty;
    Tensor shape_scalar({}, Dtype::kFloat32);
    Check(empty.empty() && !shape_scalar.empty() && shape_scalar.dim() == 0,
          "empty and shape constructors remain distinct");
}

void TestTensorFillAndFull() {
    using namespace mtensor;
    Tensor empty;
    Tensor scalar(2.5f);
    Tensor one({1}, Dtype::kFloat32);
    Check(scalar.is_scalar() && !empty.is_scalar() && !one.is_scalar(), "scalar means non-empty rank zero");
    Tensor result = full({2, 3}, scalar, Dtype::kFloat32);
    const float* data = static_cast<const float*>(result.data_ptr());
    for (int i = 0; i < 6; ++i) Check(data[i] == 2.5f, "full from scalar tensor");
    SumNode node;
    node.inputs = {result};
    const auto gradients = node.backward(scalar);
    Check(gradients.size() == 1 && gradients[0].shape() == result.shape(), "sum backward shape");
    Tensor contiguous_grad = gradients[0].clone();
    const float* grad_data = static_cast<const float*>(contiguous_grad.data_ptr());
    for (int i = 0; i < 6; ++i) Check(grad_data[i] == 2.5f, "sum backward value");
    Tensor scalar_result = full({}, 7, Dtype::kInt32);
    Check(scalar_result.is_scalar() && *static_cast<const std::int32_t*>(scalar_result.data_ptr()) == 7,
          "full with empty shape creates scalar");

    float original[] = {1, 2, 3, 4, 5, 6};
    Tensor target(original, {2, 3}, Dtype::kFloat32);
    {
        NogradNode guard;
        Tensor value = target[0][1];
        Tensor view = target.narrow(1, 1, 1);
        Check(&view.fill_(value) == &view, "tensor fill returns target");
    }
    const float* updated = static_cast<const float*>(target.data_ptr());
    const float expected[] = {1, 2, 3, 4, 2, 6};
    for (int i = 0; i < 6; ++i) Check(updated[i] == expected[i], "strided alias fill");

    std::uint16_t bits = 0x3fc1;
    Tensor bf_value(&bits, {}, Dtype::kBFloat16);
    Tensor bf_result = full({2}, bf_value, Dtype::kBFloat16);
    const auto* bf_data = static_cast<const std::uint16_t*>(bf_result.data_ptr());
    Check(bf_data[0] == bits && bf_data[1] == bits, "tensor fill preserves BF16 bits");

    bool invalid_shape = false, invalid_dtype = false, invalid_empty = false;
    try { result.fill_(one); } catch (const std::invalid_argument&) { invalid_shape = true; }
    try { result.fill_(Tensor(1)); } catch (const std::invalid_argument&) { invalid_dtype = true; }
    try { result.fill_(empty); } catch (const std::invalid_argument&) { invalid_empty = true; }
    Check(invalid_shape && invalid_dtype && invalid_empty, "invalid tensor fill values must throw");
}

}  // namespace

int main() {
    try {
        TestScalar();
        TestFill();
        TestScalarTensorConstruction();
        TestTensorFillAndFull();
        std::cout << "scalar and fill tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
