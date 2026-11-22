#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef LINEAR_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/modules/linear.h"

namespace {
using namespace mtensor;

#ifdef LINEAR_TEST_CUDA
constexpr Device kDevice = Device::kCUDA;
#else
constexpr Device kDevice = Device::kCPU;
#endif

void Check(const Tensor& tensor, const std::vector<int>& shape,
           const std::vector<float>& values, const char* name) {
    if (tensor.shape() != shape || tensor.dtype() != Dtype::kFloat32 ||
        tensor.device() != kDevice || tensor.numel() != values.size()) {
        throw std::runtime_error(name);
    }
    const Tensor host = tensor.contiguous().to(Device::kCPU);
    if (!std::equal(values.begin(), values.end(), static_cast<const float*>(host.data_ptr()))) {
        throw std::runtime_error(name);
    }
}

void CheckLinear(bool with_bias) {
    const float x_data[] = {1, 2, 3, 4, 5, 6};
    const float w_data[] = {1, 0, 2, 2, 1, -1};
    const float b_data[] = {10, -2};
    const float g_data[] = {2, 3, 5, 7};
    Tensor x(x_data, {2, 3}, Dtype::kFloat32, kDevice);
    Tensor w(w_data, {2, 3}, Dtype::kFloat32, kDevice);
    Tensor b(b_data, {2}, Dtype::kFloat32, kDevice);
    const Tensor g(g_data, {2, 2}, Dtype::kFloat32, kDevice);
    x.requires_grad_(true);
    w.requires_grad_(true);
    b.requires_grad_(true);

    Linear affine = with_bias ? Linear(w, b) : Linear(w);
    Module& module = affine;
    Tensor y = module(x);
    if (with_bias) Check(y, {2, 2}, {17, -1, 26, 5}, "linear with bias");
    else Check(y, {2, 2}, {7, 1, 16, 7}, "linear without bias");
    y.backward(g);
    Check(x.grad(), {2, 3}, {8, 3, 1, 19, 7, 3}, "linear input gradient");
    Check(w.grad(), {2, 3}, {22, 29, 36, 31, 41, 51}, "linear weight gradient");
    if (with_bias) Check(b.grad(), {2}, {7, 10}, "linear bias gradient");
}

void CheckInvalidParameters() {
    struct Case {
        std::vector<int> weight_shape, bias_shape;
        const char* name;
    };
    std::string accepted;
    for (const Case& c : {Case{{3}, {2}, "rank-1 weight"},
                          Case{{2, 3}, {2, 1}, "rank-2 bias"},
                          Case{{2, 3}, {1}, "bias length 1 instead of 2"}}) {
        const Tensor weight(c.weight_shape, Dtype::kFloat32, kDevice);
        const Tensor bias(c.bias_shape, Dtype::kFloat32, kDevice);
        try {
            Linear layer(weight, bias);
        } catch (const std::invalid_argument&) {
            continue;
        }
        if (!accepted.empty()) accepted += ", ";
        accepted += c.name;
    }
    if (!accepted.empty()) throw std::runtime_error("Linear constructor accepted: " + accepted);
}
}  // namespace

int main() {
    try {
#ifdef LINEAR_TEST_CUDA
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
            (status == cudaSuccess && count == 0)) return 77;
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
        CheckLinear(false);
        CheckLinear(true);
        CheckInvalidParameters();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
