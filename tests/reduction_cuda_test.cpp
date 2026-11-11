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

void TestBlockBoundaries() {
    for (int size : {1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 65537, 262145}) {
        CompareSum<float>(Dtype::kFloat32, std::vector<float>(size, 1.0f));
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
        TestBlockBoundaries();
        TestViewsAndBackward();
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
        std::cout << "CUDA reduction tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
