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

void TestMeanAndBroadcast() {
    const Device device = Device::kCPU;
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor input(data, {2, 3}, Dtype::kFloat32, device);
    input.requires_grad_(true);
    Tensor reduced = mean(input, -1, true);
    Tensor loss = mean(reduced);
    loss.backward();
    NogradNode guard;
    Tensor host = reduced.to(Device::kCPU);
    Check(host.shape() == std::vector<int>({2, 1}), "mean keepdim shape");
    const auto* values = static_cast<const float*>(host.data_ptr());
    Check(values[0] == 2 && values[1] == 5, "dimension mean values");
    Tensor grad = input.grad().to(Device::kCPU);
    for (int i = 0; i < 6; ++i)
        Check(std::abs(static_cast<const float*>(grad.data_ptr())[i] - 1.0f / 6) < 1e-6f,
              "mean backward");
    Tensor view = reduced.broadcast({2, 4});
    Check(view.data_ptr() == reduced.data_ptr(), "broadcast must share storage");
    Check(view.stride() == std::vector<std::size_t>({1, 0}), "broadcast stride");
    Tensor broadcasted = view.clone();
    for (int i = 0; i < 8; ++i)
        Check(static_cast<const float*>(broadcasted.data_ptr())[i] == (i < 4 ? 2 : 5),
              "broadcast repeated rows");

    auto expect_rejected = [&](auto write) {
        bool rejected = false;
        try { write(); } catch (const std::invalid_argument&) { rejected = true; }
        Check(rejected, "broadcast view write must be rejected");
    };
    Tensor scalar(1.0f);
    Tensor source({2, 4}, Dtype::kFloat32);
    expect_rejected([&] { view.fill_(1.0f); });
    expect_rejected([&] { view.fill_(scalar); });
    expect_rejected([&] { view.zero_(); });
    expect_rejected([&] { view.copy_(source); });
    source.copy_(view);
    Check(static_cast<const float*>(source.data_ptr())[4] == 5, "broadcast source copy");
}

void TestBroadcastBackward() {
    float data[] = {1, 2, 3};
    Tensor input(data, {1, 3}, Dtype::kFloat32);
    input.requires_grad_(true);
    Tensor view = input.broadcast({2, 4, 3});
    sum(view).backward();
    Tensor grad = input.grad();
    Check(grad.shape() == std::vector<int>({1, 3}), "broadcast backward shape");
    const auto* values = static_cast<const float*>(grad.data_ptr());
    for (int i = 0; i < 3; ++i) Check(values[i] == 8, "broadcast backward value");

    Tensor scalar(2.0f);
    scalar.requires_grad_(true);
    sum(scalar.broadcast({2, 3})).backward();
    Check(*static_cast<const float*>(scalar.grad().data_ptr()) == 6, "scalar broadcast backward");
}

void TestBroadcastBoundaries() {
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor base(data, {3, 1, 2}, Dtype::kFloat32);
    base.requires_grad_(true);
    Tensor input = base.transpose(0, 1);
    Tensor view = input.broadcast({2, 4, 3, 2});
    Check(view.data_ptr() == base.data_ptr(), "broadcast non-contiguous input shares storage");
    Check(view.stride() == std::vector<std::size_t>({0, 0, 2, 1}), "broadcast strides");
    float weights[48];
    for (int i = 0; i < 48; ++i) weights[i] = static_cast<float>(i + 1);
    Tensor weight(weights, {2, 4, 3, 2}, Dtype::kFloat32);
    sum(view * weight).backward();
    const auto* grad = static_cast<const float*>(base.grad().data_ptr());
    for (int j = 0; j < 6; ++j) {
        float expected = 0;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 4; ++b) expected += weights[(a * 4 + b) * 6 + j];
        Check(grad[j] == expected, "weighted broadcast gradient");
    }

    NogradNode guard;
    Tensor unchanged = input.broadcast({1, 3, 2});
    Check(unchanged.stride() == input.stride(), "unchanged singleton stride");
    bool invalid = false;
    try { input.broadcast({2, 2}); } catch (const std::invalid_argument&) { invalid = true; }
    Check(invalid, "broadcast rank check");
    invalid = false;
    try { input.broadcast({2, 4, 2}); } catch (const std::invalid_argument&) { invalid = true; }
    Check(invalid, "broadcast incompatible shape");
    invalid = false;
    try { input.broadcast({-1, 3, 2}); } catch (const std::invalid_argument&) { invalid = true; }
    Check(invalid, "broadcast negative dimension");
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

void TestDimensionSum() {
    NogradNode guard;
    float values[24];
    for (int i = 0; i < 24; ++i) values[i] = static_cast<float>(i + 1);
    Tensor input(values, {2, 3, 4}, Dtype::kFloat32);
    Tensor result = sum(input, 1);
    Check(result.shape() == std::vector<int>({2, 4}), "dimension sum shape");
    const float expected[] = {15, 18, 21, 24, 51, 54, 57, 60};
    for (int i = 0; i < 8; ++i)
        Check(static_cast<const float*>(result.data_ptr())[i] == expected[i], "dimension sum value");
    Check(sum(input, -2, true).shape() == std::vector<int>({2, 1, 4}), "keepdim shape");
    Tensor view_result = sum(input.transpose(0, 1), 0);
    for (int i = 0; i < 8; ++i)
        Check(static_cast<const float*>(view_result.data_ptr())[i] == expected[i], "strided dimension sum");
    for (int dim : {0, 1, 2}) {
        Tensor reduced = sum(input, dim);
        Check(*static_cast<const float*>(sum(reduced).data_ptr()) == 300, "axis sum total");
    }
    bool threw = false;
    try { sum(input, 3); } catch (const std::out_of_range&) { threw = true; }
    Check(threw, "invalid dimension");
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
        TestMeanAndBroadcast();
        TestBroadcastBackward();
        TestBroadcastBoundaries();
        TestDimensionSum();
        TestViewsAndBackward();
        std::cout << "CPU reduction tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
