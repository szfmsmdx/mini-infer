#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "autograd/autograd.h"
#include "autograd/backward_nodes/tensor_ops.h"
#include "ops/elementwise.h"
#include "tensor/tensor.h"

namespace {
using namespace mtensor;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckScalarGradient(const Tensor& tensor, float expected) {
    Check(tensor.has_grad(), "missing gradient");
    NogradNode guard;
    Tensor cpu_grad = tensor.grad().to(Device::kCPU);
    float actual = *static_cast<const float*>(cpu_grad.data_ptr());
    Check(std::fabs(actual - expected) < 1e-5f, "incorrect gradient value");
    Check(tensor.grad().device() == tensor.device(), "gradient is on wrong device");
}

void TestTransfer(Device source, Device destination) {
    float value = 3.0f;
    Tensor x(&value, {}, Dtype::kFloat32, source);
    x.requires_grad_(true);
    Tensor y = x.to(destination);
    Check(y.grad_node() != nullptr, "transfer must record a node");
    Check(y.to(destination).grad_node() == y.grad_node(), "same-device to must share metadata");
    Tensor loss = mul(y, y);
    loss.backward();
    CheckScalarGradient(x, 6.0f);
    Check(EnableGrad(), "backward must restore gradient recording");
}

void TestTransferChain() {
    float value = 2.0f;
    Tensor x(&value, {}, Dtype::kFloat32);
    x.requires_grad_(true);
    Tensor y = mul(x, x).to(Device::kCUDA).to(Device::kCPU);
    y.backward();
    CheckScalarGradient(x, 4.0f);
}

void TestNoGradAndInteger() {
    float value = 1.0f;
    Tensor x(&value, {}, Dtype::kFloat32);
    {
        NogradNode guard;
        Check(!x.to(Device::kCUDA).grad_node(), "no-grad transfer must not record a node");
    }
    int integer = 1;
    Tensor x_int(&integer, {}, Dtype::kInt32);
    Check(!x_int.to(Device::kCUDA).grad_node(), "integer transfer must not record a node");
}

void TestNonContiguousTransfer() {
    float values[] = {1, 2, 3, 4};
    Tensor x(values, {2, 2}, Dtype::kFloat32);
    Tensor view;
    {
        NogradNode guard;
        view = x.narrow(1, 0, 1);
    }
    view.requires_grad_(true);
    Tensor y = view.to(Device::kCUDA);
    auto transfer = y.grad_node();
    Check(dynamic_cast<ToDeviceNode*>(transfer.get()) != nullptr, "missing transfer node");
    Check(dynamic_cast<CloneNode*>(transfer->inputs[0].grad_node().get()) != nullptr,
          "non-contiguous transfer must retain clone node");
    Tensor seed(y.shape(), y.dtype(), y.device());
    seed.fill_(1);
    NogradNode guard;
    auto grads = transfer->backward(seed);
    Check(grads.size() == 1 && grads[0].device() == Device::kCPU &&
          grads[0].shape() == view.shape(), "non-contiguous transfer gradient metadata");
}
}  // namespace

int main() {
    try {
        int count = 0;
        cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) return 77;
        Check(status == cudaSuccess, "cudaGetDeviceCount failed");
        TestTransfer(Device::kCPU, Device::kCUDA);
        TestTransfer(Device::kCUDA, Device::kCPU);
        TestTransferChain();
        TestNoGradAndInteger();
        TestNonContiguousTransfer();
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
        std::cout << "CUDA autograd tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
