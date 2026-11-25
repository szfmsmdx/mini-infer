#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mini_infer/autograd.h"
#include "mini_infer/tensor/tensor.h"

namespace {
using mtensor::Device;
using mtensor::Dtype;
using mtensor::Tensor;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Make>
void CheckRankRejected(Make make) {
    try {
        (void)make();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("rank 9 must be rejected");
}

void RunTests() {
    float values[] = {0, 1, 2, 3, 4, 5};
    const std::vector<int> shape8{2, 1, 1, 1, 1, 1, 1, 3};
    const std::vector<int> shape9{1, 2, 1, 1, 1, 1, 1, 1, 3};
    Tensor wide(values, shape8, Dtype::kFloat32, Device::kCUDA);
    Tensor strided = wide.transpose(0, 7);
    const Tensor host_wide = (strided + strided).to(Device::kCPU);
    Check(host_wide.dim() == 8 && !strided.is_contiguous(), "rank 8 strided test setup");
    const auto* high_rank = static_cast<const float*>(host_wide.data_ptr());
    Check(high_rank[0] == 0 && high_rank[1] == 6 && high_rank[2] == 2 &&
          high_rank[3] == 8 && high_rank[4] == 4 && high_rank[5] == 10,
          "rank 8 CUDA layout passed to kernels incorrectly");
    CheckRankRejected([&] { return Tensor(shape9, Dtype::kFloat32, Device::kCUDA); });
    CheckRankRejected([&] { return wide.unsqueeze(0); });
    CheckRankRejected([&] { return wide.reshape(shape9); });
    CheckRankRejected([&] { return wide.broadcast(shape9); });

    Tensor source(values, {2, 3}, Dtype::kFloat32, Device::kCUDA);
    Tensor columns = source.narrow(1, 1, 2).transpose(0, 1);
    Tensor tail = source[1].narrow(0, 1, 2).unsqueeze(0);
    Check(columns.stride() == std::vector<std::size_t>({1, 3}), "transpose stride");
    Check(tail.shape() == std::vector<int>({1, 2}), "nested view shape");
    source = Tensor();

    const Tensor host_columns = (columns + columns).to(Device::kCPU);
    const auto* doubled = static_cast<const float*>(host_columns.data_ptr());
    Check(doubled[0] == 2 && doubled[1] == 8 && doubled[2] == 4 && doubled[3] == 10,
          "strided CUDA layout passed to kernels incorrectly");
    const Tensor host_tail = tail.clone().to(Device::kCPU);
    const auto* tail_values = static_cast<const float*>(host_tail.data_ptr());
    Check(tail_values[0] == 4 && tail_values[1] == 5, "nested CUDA view lost its offset");
    const Tensor scalar = tail[0][1] + tail[0][1];
    const Tensor host_scalar = scalar.to(Device::kCPU);
    Check(host_scalar.is_scalar() && *static_cast<const float*>(host_scalar.data_ptr()) == 10,
          "rank zero CUDA layout passed to kernels incorrectly");
}
}  // namespace

int main() {
    try {
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) return 77;
        Check(status == cudaSuccess, "cudaGetDeviceCount failed");
        mtensor::NogradNode guard;
        RunTests();
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
        std::cout << "CUDA layout tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
