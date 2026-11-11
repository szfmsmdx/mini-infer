#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "autograd/autograd.h"
#include "ops/elementwise.h"
#include "ops/reduction.h"
#include "tensor/tensor.h"

namespace {
using namespace mtensor;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void CheckIntegerSum(Dtype dtype, const std::vector<T>& values, std::int64_t expected) {
    Tensor input(values.data(), {static_cast<int>(values.size())}, dtype);
    Tensor result = sum(input);
    Check(result.is_scalar() && result.dtype() == Dtype::kInt64, "integer sum metadata");
    Check(*static_cast<const std::int64_t*>(result.data_ptr()) == expected, "integer sum value");
    Check(!result.grad_node(), "integer sum must not build a gradient node");
}

void TestDtypes() {
    CheckIntegerSum<std::int8_t>(Dtype::kInt8, {-2, 3, 4}, 5);
    CheckIntegerSum<std::uint8_t>(Dtype::kUInt8, {200, 100}, 300);
    CheckIntegerSum<std::int16_t>(Dtype::kInt16, {-30000, 100, 10}, -29890);
    CheckIntegerSum<std::int32_t>(Dtype::kInt32, {2000000000, 2000000000}, 4000000000LL);
    CheckIntegerSum<std::int64_t>(Dtype::kInt64, {(std::int64_t{1} << 60) + 3, -2},
                                  (std::int64_t{1} << 60) + 1);
    CheckIntegerSum<std::int64_t>(Dtype::kInt64, {std::numeric_limits<std::int64_t>::max(), 1},
                                  std::numeric_limits<std::int64_t>::min());

    const float values[] = {1.0f, -2.0f, 4.5f};
    Tensor fp32(values, {3}, Dtype::kFloat32);
    Tensor result = sum(fp32);
    Check(result.is_scalar() && result.dtype() == Dtype::kFloat32, "Float32 sum metadata");
    Check(*static_cast<const float*>(result.data_ptr()) == 3.5f, "Float32 sum value");

    for (Dtype dtype : {Dtype::kFloat16, Dtype::kBFloat16}) {
        std::uint16_t encoded[3];
        for (int i = 0; i < 3; ++i) {
            encoded[i] = dtype == Dtype::kFloat16 ? EncodeFloat16(values[i]) : EncodeBFloat16(values[i]);
        }
        Tensor input(encoded, {3}, dtype);
        Tensor half_result = sum(input);
        Check(half_result.is_scalar() && half_result.dtype() == dtype, "half sum metadata");
        const float actual = dtype == Dtype::kFloat16 ? DecodeFloat16(half_result.data_ptr())
                                                      : DecodeBFloat16(half_result.data_ptr());
        Check(actual == 3.5f, "half sum value");
        input.requires_grad_(true);
        Tensor loss = sum(input);
        loss.backward();
        Check(input.has_grad(), "missing half gradient");
        const auto* grad = static_cast<const std::uint16_t*>(input.grad().data_ptr());
        const std::uint16_t one = dtype == Dtype::kFloat16 ? EncodeFloat16(1.0f) : EncodeBFloat16(1.0f);
        for (int i = 0; i < 3; ++i) Check(grad[i] == one, "half sum gradient");
    }

    const float inf_values[] = {std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()};
    Tensor inf_input(inf_values, {2}, Dtype::kFloat32);
    Check(std::isnan(*static_cast<const float*>(sum(inf_input).data_ptr())), "NaN propagation");
}

void TestViewsAndBackward() {
    float values[] = {1, 2, 3, 4, 5, 6};
    Tensor input(values, {2, 3}, Dtype::kFloat32);
    input.requires_grad_(true);
    Tensor transposed = input.transpose(0, 1);
    Tensor loss = sum(mul(transposed, transposed));
    Check(*static_cast<const float*>(loss.data_ptr()) == 91.0f, "strided sum value");
    loss.backward();
    Check(input.has_grad() && input.grad().shape() == input.shape(), "gradient shape");
    const float* grad = static_cast<const float*>(input.grad().data_ptr());
    for (int i = 0; i < 6; ++i) Check(grad[i] == 2 * values[i], "gradient through transpose");

    {
        NogradNode guard;
        Tensor view = input.narrow(1, 1, 1);
        Check(*static_cast<const float*>(sum(view).data_ptr()) == 7.0f, "narrow view sum");
        Check(!sum(view).grad_node(), "no-grad sum must not build a node");
    }
    Tensor scalar(2.5f);
    Check(*static_cast<const float*>(sum(scalar).data_ptr()) == 2.5f, "scalar sum");
    bool threw = false;
    try { sum(Tensor()); } catch (const std::invalid_argument&) { threw = true; }
    Check(threw, "empty tensor must be rejected");
}
}  // namespace

int main() {
    try {
        TestDtypes();
        TestViewsAndBackward();
        std::cout << "CPU reduction tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
