#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
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
        TestMeanAndBroadcast();
        TestBroadcastBackward();
        TestBlockBoundaries();
        TestDimensionSum();
        TestViewsAndBackward();
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
        std::cout << "CUDA reduction tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
