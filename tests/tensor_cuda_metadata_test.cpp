#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/tensor.h"

namespace mtensor {
struct TensorMetadataTestAccess {
    static const void* shape(const Tensor& tensor) { return tensor.shape_cuda_.get(); }
    static const void* stride(const Tensor& tensor) { return tensor.stride_cuda_.get(); }
};
}

namespace {
using mtensor::Device;
using mtensor::Dtype;
using mtensor::Tensor;
using Access = mtensor::TensorMetadataTestAccess;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckMetadata(const Tensor& tensor) {
    Check(Access::shape(tensor) != nullptr && Access::stride(tensor) != nullptr,
          "CUDA metadata pointers must be present");
    std::vector<int> shape(tensor.shape().size());
    std::vector<std::size_t> stride(tensor.stride().size());
    Check(cudaMemcpy(shape.data(), Access::shape(tensor), shape.size() * sizeof(int),
                     cudaMemcpyDeviceToHost) == cudaSuccess, "shape device copy failed");
    Check(cudaMemcpy(stride.data(), Access::stride(tensor), stride.size() * sizeof(std::size_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess, "stride device copy failed");
    Check(shape == tensor.shape() && stride == tensor.stride(), "CUDA metadata differs from host");
}

void CheckNoMetadata(const Tensor& tensor) {
    Check(Access::shape(tensor) == nullptr && Access::stride(tensor) == nullptr,
          "metadata pointers should be empty");
}

void RunTests() {
    float values[] = {0, 1, 2, 3, 4, 5};
    Tensor source(values, {2, 3}, Dtype::kFloat32, Device::kCUDA);
    Tensor allocated({3, 2}, Dtype::kFloat32, Device::kCUDA);
    CheckMetadata(source);
    CheckMetadata(allocated);
    Tensor cpu(values, {2, 3}, Dtype::kFloat32);
    CheckNoMetadata(cpu);
    Tensor scalar(values, {}, Dtype::kFloat32, Device::kCUDA);
    Tensor allocated_scalar({}, Dtype::kFloat32, Device::kCUDA);
    CheckNoMetadata(scalar);
    CheckNoMetadata(allocated_scalar);

    Tensor copied(source);
    Check(Access::shape(copied) == Access::shape(source) &&
          Access::stride(copied) == Access::stride(source), "copy must share metadata");
    allocated = source;
    Check(Access::shape(allocated) == Access::shape(source) &&
          Access::stride(allocated) == Access::stride(source), "copy assignment must share metadata");
    source = Tensor();
    CheckMetadata(copied);
    CheckMetadata(allocated);

    const void* shape = Access::shape(copied);
    const void* stride = Access::stride(copied);
    Tensor moved(std::move(copied));
    Check(Access::shape(moved) == shape && Access::stride(moved) == stride,
          "move constructor must transfer metadata");
    CheckNoMetadata(copied);
    CheckMetadata(moved);

    Tensor destination(values, {3, 2}, Dtype::kFloat32, Device::kCUDA);
    destination = std::move(moved);
    Check(Access::shape(destination) == shape && Access::stride(destination) == stride,
          "move assignment must transfer metadata");
    CheckNoMetadata(moved);
    CheckMetadata(destination);
    destination = destination;
    destination = std::move(destination);
    CheckMetadata(destination);
    destination = cpu;
    CheckNoMetadata(destination);
    allocated = Tensor();

    Tensor gpu = cpu.to(Device::kCUDA);
    CheckMetadata(gpu);
    Tensor same_device = gpu.to(Device::kCUDA);
    Check(Access::shape(same_device) == Access::shape(gpu) &&
          Access::stride(same_device) == Access::stride(gpu), "same-device to must share metadata");
    Tensor host = gpu.to(Device::kCPU);
    CheckNoMetadata(host);
    Check(host.shape() == cpu.shape(), "device transfer must preserve shape");

    const void* original_shape = Access::shape(gpu);
    const void* original_stride = Access::stride(gpu);
    Tensor clone = gpu.clone();
    CheckMetadata(clone);
    Check(clone.data_ptr() != gpu.data_ptr(), "clone must have independent data");
    Check(Access::shape(gpu) == original_shape && Access::stride(gpu) == original_stride,
          "clone must not modify source metadata");
    CheckMetadata(gpu);

    Tensor columns = gpu.narrow(1, 1, 2);
    CheckMetadata(columns);
    Check(columns.shape() == std::vector<int>({2, 2}) &&
          columns.stride() == std::vector<std::size_t>({3, 1}), "narrow layout is wrong");
    Tensor row = columns[1];
    CheckMetadata(row);
    Check(row.shape() == std::vector<int>({2}) && row.stride() == std::vector<std::size_t>({1}),
          "indexed view layout is wrong");
    Tensor scalar_view = row[1];
    CheckNoMetadata(scalar_view);
    Tensor reshaped = gpu.reshape({3, 2});
    CheckMetadata(reshaped);
    Check(reshaped.shape() == std::vector<int>({3, 2}) &&
          reshaped.stride() == std::vector<std::size_t>({2, 1}), "reshape layout is wrong");
    CheckMetadata(gpu);
    gpu = Tensor();
    same_device = Tensor();
    CheckMetadata(columns);
    CheckMetadata(row);
    CheckMetadata(reshaped);

    Tensor cpu_scalar(values, {}, Dtype::kFloat32);
    Tensor gpu_scalar = cpu_scalar.to(Device::kCUDA);
    CheckNoMetadata(gpu_scalar);
    CheckNoMetadata(gpu_scalar.clone());
    Tensor empty_gpu = Tensor().to(Device::kCUDA);
    Check(empty_gpu.empty() && empty_gpu.device() == Device::kCUDA, "empty migration failed");
    CheckNoMetadata(empty_gpu);
    CheckNoMetadata(empty_gpu.clone());
}
}

int main() {
    try {
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) return 77;
        Check(status == cudaSuccess, "cudaGetDeviceCount failed");
        RunTests();
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
        std::cout << "CUDA metadata tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
