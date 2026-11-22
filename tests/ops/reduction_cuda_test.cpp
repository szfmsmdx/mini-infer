#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mini_infer/autograd.h"
#include "mini_infer/ops/elementwise.h"
#include "mini_infer/ops/reduction.h"
#include "mini_infer/tensor.h"
#include "reduction_extrema.h"

namespace {
using namespace mtensor;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void CompareSum(Dtype dtype, const std::vector<T>& data) {
    Tensor cpu(data.data(), {static_cast<int>(data.size())}, dtype);
    Tensor gpu(data.data(), {static_cast<int>(data.size())}, dtype, Device::kCUDA);
    NogradNode guard;
    Tensor expected = sum(cpu);
    Tensor actual = sum(gpu).to(Device::kCPU);
    Check(actual.is_scalar() && actual.dtype() == expected.dtype(), "CUDA sum metadata");
    Check(std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.nbytes()) == 0,
          "CUDA sum differs from CPU");
}

void TestAddBroadcast() {
    float a_data[] = {1, 2};
    float b_data[] = {10, 20, 30};
    Tensor a(a_data, {2, 1}, Dtype::kFloat32, Device::kCUDA);
    Tensor b(b_data, {3}, Dtype::kFloat32, Device::kCUDA);
    a.requires_grad_(true);
    b.requires_grad_(true);
    Tensor result = add(a, b);
    Check(result.shape() == std::vector<int>({2, 3}), "add broadcast shape");
    sum(result).backward();
    NogradNode guard;
    Tensor host = result.to(Device::kCPU);
    const auto* values = static_cast<const float*>(host.data_ptr());
    for (int i = 0; i < 6; ++i) {
        Check(values[i] == a_data[i / 3] + b_data[i % 3], "add broadcast values");
    }
    Tensor ga = a.grad().to(Device::kCPU);
    Tensor gb = b.grad().to(Device::kCPU);
    Check(ga.shape() == a.shape() && gb.shape() == b.shape(), "add broadcast gradient shapes");
    for (int i = 0; i < 2; ++i)
        Check(static_cast<const float*>(ga.data_ptr())[i] == 3, "add lhs gradient");
    for (int i = 0; i < 3; ++i)
        Check(static_cast<const float*>(gb.data_ptr())[i] == 2, "add rhs gradient");

    Tensor scalar = full({}, 5.0f, Dtype::kFloat32, Device::kCUDA);
    Tensor strided = result.transpose(0, 1);
    Tensor shifted = add(scalar, strided).to(Device::kCPU);
    Tensor reversed = add(strided, scalar).to(Device::kCPU);
    for (int i = 0; i < 6; ++i) {
        const float expected = a_data[i % 2] + b_data[i / 2] + 5;
        Check(static_cast<const float*>(shifted.data_ptr())[i] == expected, "scalar strided add");
        Check(static_cast<const float*>(reversed.data_ptr())[i] == expected, "reversed broadcast add");
    }
    Check(!shifted.grad_node(), "no-grad broadcast add");
    bool rejected = false;
    try { add(b, full({2}, 0.0f, Dtype::kFloat32, Device::kCUDA)); }
    catch (const std::invalid_argument&) { rejected = true; }
    Check(rejected, "incompatible add broadcast shapes");
}
void TestBinaryBroadcast() {
    using BinaryOp = Tensor (*)(const Tensor&, const Tensor&);
    const BinaryOp operations[] = {sub, mul, div};
    for (int op = 0; op < 3; ++op) {
        float a_data[] = {2, 4};
        float b_data[] = {1, 2, 4};
        Tensor a(a_data, {2, 1}, Dtype::kFloat32, Device::kCUDA);
        Tensor b(b_data, {3}, Dtype::kFloat32, Device::kCUDA);
        a.requires_grad_(true);
        b.requires_grad_(true);
        Tensor result = operations[op](a, b);
        Check(result.shape() == std::vector<int>({2, 3}), "binary broadcast shape");
        sum(result).backward();
        NogradNode guard;
        Tensor host = result.to(Device::kCPU);
        for (int i = 0; i < 6; ++i) {
            const float av = a_data[i / 3], bv = b_data[i % 3];
            const float expected = op == 0 ? av - bv : (op == 1 ? av * bv : av / bv);
            Check(static_cast<const float*>(host.data_ptr())[i] == expected, "binary broadcast value");
        }
        Tensor ga = a.grad().to(Device::kCPU), gb = b.grad().to(Device::kCPU);
        Check(ga.shape() == a.shape() && gb.shape() == b.shape(), "binary broadcast grad shapes");
        for (int i = 0; i < 2; ++i) {
            const float expected = op == 0 ? 3.0f : (op == 1 ? 7.0f : 1.75f);
            Check(static_cast<const float*>(ga.data_ptr())[i] == expected, "binary broadcast lhs grad");
        }
        for (int i = 0; i < 3; ++i) {
            const float expected = op == 0 ? -2.0f : (op == 1 ? 6.0f : -6.0f / (b_data[i] * b_data[i]));
            Check(static_cast<const float*>(gb.data_ptr())[i] == expected, "binary broadcast rhs grad");
        }
        Tensor scalar = full({}, 2.0f, Dtype::kFloat32, Device::kCUDA);
        Tensor view = result.transpose(0, 1);
        for (bool reverse : {false, true}) {
            Tensor output = reverse ? operations[op](scalar, view) : operations[op](view, scalar);
            Check(!output.grad_node(), "no-grad binary broadcast");
            Tensor actual = output.to(Device::kCPU);
            for (int i = 0; i < 6; ++i) {
                const float value = static_cast<const float*>(host.data_ptr())[(i % 2) * 3 + i / 2];
                const float lhs = reverse ? 2.0f : value, rhs = reverse ? value : 2.0f;
                const float expected = op == 0 ? lhs - rhs : (op == 1 ? lhs * rhs : lhs / rhs);
                Check(static_cast<const float*>(actual.data_ptr())[i] == expected, "scalar strided binary broadcast");
            }
        }
        bool rejected = false;
        try { operations[op](b, full({2}, 1.0f, Dtype::kFloat32, Device::kCUDA)); }
        catch (const std::invalid_argument&) { rejected = true; }
        Check(rejected, "incompatible binary broadcast shapes");
    }
}
void TestBroadcastDtypesAndRanks() {
    using BinaryOp = Tensor (*)(const Tensor&, const Tensor&);
    const BinaryOp operations[] = {add, sub, mul, div};
    for (Dtype dtype : {Dtype::kFloat32, Dtype::kFloat16, Dtype::kBFloat16, Dtype::kInt32}) {
        for (int op = 0; op < 4; ++op) {
            // Three leading dimensions on rhs; both inputs expand along different axes.
            Tensor a = full({2, 1, 3, 1}, 8, dtype, Device::kCUDA);
            Tensor b = full({4}, 2, dtype, Device::kCUDA);
            if (dtype != Dtype::kInt32) {
                a.requires_grad_(true);
                b.requires_grad_(true);
            }
            Tensor result = operations[op](a, b);
            Check(result.shape() == std::vector<int>({2, 1, 3, 4}), "multi-axis broadcast shape");
            const int expected = op == 0 ? 10 : (op == 1 ? 6 : (op == 2 ? 16 : 4));
            if (dtype != Dtype::kInt32) sum(result).backward();
            else Check(!result.grad_node(), "integer broadcast must not build graph");
            NogradNode guard;
            Tensor host = result.to(Device::kCPU);
            auto check_value = [dtype](const Tensor& tensor, int index, float value) {
                const void* data = tensor.data_ptr();
                if (dtype == Dtype::kFloat32)
                    Check(static_cast<const float*>(data)[index] == value, "float broadcast value");
                else if (dtype == Dtype::kInt32) {
                    const auto* actual = static_cast<const std::int32_t*>(data);
                    const int expected = static_cast<int>(value);
                    Check(actual[index] == expected, "integer broadcast value");
                } else {
                    const auto bits = static_cast<const std::uint16_t*>(data)[index];
                    const auto expected_bits = dtype == Dtype::kFloat16 ? EncodeFloat16(value) : EncodeBFloat16(value);
                    Check(bits == expected_bits, "half broadcast value");
                }
            };
            for (int i = 0; i < 24; ++i) check_value(host, i, static_cast<float>(expected));
            if (dtype != Dtype::kInt32) {
                Tensor ga = a.grad().to(Device::kCPU), gb = b.grad().to(Device::kCPU);
                Check(ga.shape() == a.shape() && gb.shape() == b.shape(), "multi-axis grad shape");
                const float da = op == 2 ? 8.0f : (op == 3 ? 2.0f : 4.0f);
                const float db = op == 0 ? 6.0f : (op == 1 ? -6.0f : (op == 2 ? 48.0f : -12.0f));
                for (int i = 0; i < 6; ++i) check_value(ga, i, da);
                for (int i = 0; i < 4; ++i) check_value(gb, i, db);
            }
        }
    }
    // The same leaf reaches the loss through a direct path and a broadcast path.
    Tensor scalar = full({}, 2.0f, Dtype::kFloat32, Device::kCUDA);
    scalar.requires_grad_(true);
    Tensor expanded = scalar.broadcast({2, 3});
    sum(mul(scalar, expanded)).backward();
    NogradNode guard;
    Tensor grad = scalar.grad().to(Device::kCPU);
    Check(grad.is_scalar() && *static_cast<const float*>(grad.data_ptr()) == 24,
          "shared leaf broadcast gradient accumulation");
}
void TestMeanAndBroadcast() {
    const Device device = Device::kCUDA;
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
    Tensor broadcasted = reduced.broadcast({2, 4}).to(Device::kCPU);
    for (int i = 0; i < 8; ++i)
        Check(static_cast<const float*>(broadcasted.data_ptr())[i] == (i < 4 ? 2 : 5),
              "broadcast repeated rows");
}

void TestDtypes() {
    CompareSum<std::int8_t>(Dtype::kInt8, {-2, 3, 4});
    CompareSum<std::uint8_t>(Dtype::kUInt8, {200, 100});
    CompareSum<std::int16_t>(Dtype::kInt16, {-30000, 100, 10});
    CompareSum<std::int32_t>(Dtype::kInt32, {2000000000, 2000000000});
    CompareSum<std::int64_t>(Dtype::kInt64, {std::numeric_limits<std::int64_t>::max(), 1});
    CompareSum<float>(Dtype::kFloat32, {1.0f, -2.0f, 4.5f});
    CompareSum<std::uint16_t>(Dtype::kFloat16,
        {EncodeFloat16(1.0f), EncodeFloat16(-2.0f), EncodeFloat16(4.5f)});
    CompareSum<std::uint16_t>(Dtype::kBFloat16,
        {EncodeBFloat16(1.0f), EncodeBFloat16(-2.0f), EncodeBFloat16(4.5f)});
}

void TestBroadcastBackward() {
    float data[] = {1, 2, 3, 4, 5, 6};
    float weights[48];
    for (int i = 0; i < 48; ++i) weights[i] = static_cast<float>(i + 1);
    Tensor base(data, {3, 1, 2}, Dtype::kFloat32, Device::kCUDA);
    base.requires_grad_(true);
    Tensor view = base.transpose(0, 1).broadcast({2, 4, 3, 2});
    Check(view.data_ptr() == base.data_ptr(), "CUDA broadcast must share storage");
    Check(view.stride() == std::vector<std::size_t>({0, 0, 2, 1}), "CUDA broadcast stride");
    Tensor weight(weights, {2, 4, 3, 2}, Dtype::kFloat32, Device::kCUDA);
    sum(view * weight).backward();
    NogradNode guard;
    Tensor host_grad = base.grad().to(Device::kCPU);
    const auto* grad = static_cast<const float*>(host_grad.data_ptr());
    for (int j = 0; j < 6; ++j) {
        float expected = 0;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 4; ++b) expected += weights[(a * 4 + b) * 6 + j];
        Check(grad[j] == expected, "CUDA weighted broadcast gradient");
    }
    bool rejected = false;
    try { view.zero_(); } catch (const std::invalid_argument&) { rejected = true; }
    Check(rejected, "CUDA broadcast view write must be rejected");
}

void TestBlockBoundaries() {
    for (int size : {1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 65537, 262145}) {
        CompareSum<float>(Dtype::kFloat32, std::vector<float>(size, 1.0f));
    }
}

void TestDimensionSum() {
    NogradNode guard;
    for (int length : {1, 33, 1025}) {
        std::vector<float> data(2 * length * 4);
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i % 7);
        Tensor cpu(data.data(), {2, length, 4}, Dtype::kFloat32);
        Tensor gpu = cpu.to(Device::kCUDA);
        for (bool transpose : {false, true}) {
            Tensor a = transpose ? cpu.transpose(0, 1) : cpu;
            Tensor b = transpose ? gpu.transpose(0, 1) : gpu;
            for (int dim : {0, 1, -1}) {
                for (bool keepdim : {false, true}) {
                    Tensor expected = sum(a, dim, keepdim);
                    Tensor actual = sum(b, dim, keepdim).to(Device::kCPU);
                    Check(actual.shape() == expected.shape(), "dimension sum shape");
                    Check(actual.dtype() == expected.dtype(), "dimension sum dtype");
                    Check(std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.nbytes()) == 0,
                          "CUDA dimension sum differs from CPU");
                }
            }
        }
    }
}

void TestViewsAndBackward() {
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor input(data, {2, 3}, Dtype::kFloat32, Device::kCUDA);
    input.requires_grad_(true);
    Tensor view = input.transpose(0, 1);
    Tensor loss = sum(mul(view, view));
    loss.backward();
    Check(input.has_grad(), "missing CUDA sum gradient");
    NogradNode guard;
    Tensor host_loss = loss.to(Device::kCPU);
    Tensor host_grad = input.grad().to(Device::kCPU);
    Check(*static_cast<const float*>(host_loss.data_ptr()) == 91.0f, "CUDA sum loss");
    const float* grad = static_cast<const float*>(host_grad.data_ptr());
    for (int i = 0; i < 6; ++i) Check(grad[i] == 2 * data[i], "CUDA sum gradient");

    Tensor view_sum = sum(view);
    Tensor host_sum = view_sum.to(Device::kCPU);
    Check(*static_cast<const float*>(host_sum.data_ptr()) == 21.0f, "CUDA strided sum");
}
}  // namespace

int main() {
    try {
        int count = 0;
        cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) return 77;
        Check(status == cudaSuccess, "cudaGetDeviceCount failed");
        TestDtypes();
        TestAddBroadcast();
        TestBinaryBroadcast();
        TestBroadcastDtypesAndRanks();
        TestMeanAndBroadcast();
        TestBroadcastBackward();
        TestBlockBoundaries();
        TestDimensionSum();
        mtensor::test::CheckExtrema(mtensor::Device::kCUDA);
        TestViewsAndBackward();
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
        std::cout << "CUDA reduction tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
