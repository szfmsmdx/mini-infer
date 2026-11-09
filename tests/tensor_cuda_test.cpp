#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensor/tensor.h"
#include "utils/memory.h"

namespace {

using mtensor::Device;
using mtensor::Dtype;
using mtensor::Tensor;

static_assert(noexcept(mtensor::FreeCUDA(nullptr)),
              "FreeCUDA must not throw from a shared_ptr deleter");

void Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T>
std::vector<T> Read(Tensor& tensor) {
    std::vector<T> result(tensor.numel());
    if (!result.empty()) {
        CheckCuda(cudaMemcpy(result.data(), tensor.data_ptr(), tensor.nbytes(),
                             cudaMemcpyDeviceToHost));
    }
    return result;
}

template <typename T>
void CheckValues(Tensor& tensor, const std::vector<T>& expected) {
    Check(Read<T>(tensor) == expected, "CUDA Tensor elements differ from expected values");
}

std::string Printed(const Tensor& tensor) {
    std::ostringstream output;
    std::streambuf* original = std::cout.rdbuf(output.rdbuf());
    try {
        tensor.print();
    } catch (...) {
        std::cout.rdbuf(original);
        throw;
    }
    std::cout.rdbuf(original);
    return output.str();
}

void TestMemory() {
    Check(mtensor::AllocateCUDA(0) == nullptr, "Zero-byte allocation must be null");
    mtensor::FreeCUDA(nullptr);
    void* device_ptr = mtensor::AllocateCUDA(sizeof(std::int32_t));
    Check(device_ptr != nullptr, "CUDA allocation returned null");
    const std::int32_t input = 42;
    std::int32_t output = 0;
    CheckCuda(cudaMemcpy(device_ptr, &input, sizeof(input), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(&output, device_ptr, sizeof(output), cudaMemcpyDeviceToHost));
    Check(output == input, "Allocated CUDA memory could not round-trip data");
    mtensor::FreeCUDA(device_ptr);
}

template <typename T>
void TestConstructionAndView(Dtype dtype) {
    T input[] = {1, 2, 3, 4};
    Tensor tensor(input, {2, 2}, dtype, Device::kCUDA);
    Check(!tensor.empty() && tensor.device() == Device::kCUDA && tensor.dtype() == dtype,
          "CUDA constructor lost metadata");
    Check(tensor.shape() == std::vector<int>({2, 2}) && tensor.size(1) == 2 &&
          tensor.dim() == 2 && tensor.numel() == 4 &&
          tensor.element_size() == sizeof(T) && tensor.nbytes() == sizeof(input),
          "CUDA Tensor shape or byte count is wrong");
    CheckValues<T>(tensor, {1, 2, 3, 4});
    input[0] = 99;
    CheckValues<T>(tensor, {1, 2, 3, 4});

    Tensor row = tensor[1];
    Check(row.shape() == std::vector<int>({2}) && row.device() == Device::kCUDA &&
          row.dtype() == dtype && row.data_ptr() == static_cast<T*>(tensor.data_ptr()) + 2,
          "CUDA view metadata or offset is wrong");
    CheckValues<T>(row, {3, 4});
    Tensor scalar = row[1];
    Check(scalar.shape().empty() && scalar.dim() == 0 && scalar.numel() == 1,
          "CUDA scalar view is wrong");
    CheckValues<T>(scalar, {4});
    tensor = Tensor();
    row = Tensor();
    CheckValues<T>(scalar, {4});
}

void TestConstructionAndViews() {
    TestConstructionAndView<float>(Dtype::kFloat32);
    TestConstructionAndView<std::uint16_t>(Dtype::kFloat16);
    TestConstructionAndView<std::uint16_t>(Dtype::kBFloat16);
    TestConstructionAndView<std::int8_t>(Dtype::kInt8);
    TestConstructionAndView<std::uint8_t>(Dtype::kUInt8);
    TestConstructionAndView<std::int16_t>(Dtype::kInt16);
    TestConstructionAndView<std::int32_t>(Dtype::kInt32);
    TestConstructionAndView<std::int64_t>(Dtype::kInt64);

    float input = 6;
    Tensor scalar(&input, {}, Dtype::kFloat32, Device::kCUDA);
    CheckValues<float>(scalar, {6});
    Check(scalar.numel() == 1 && scalar.nbytes() == sizeof(float),
          "Constructed CUDA scalar has wrong size");
}

void TestOwnership() {
    float input[] = {1, 2, 3, 4};
    Tensor source(input, {2, 2}, Dtype::kFloat32, Device::kCUDA);
    Tensor copied(source);
    Check(copied.data_ptr() == source.data_ptr(), "CUDA copy constructor must share storage");
    copied[0].fill_(7.0f);
    CheckValues<float>(source, {7, 7, 3, 4});

    Tensor assigned;
    Check(&(assigned = source) == &assigned && assigned.data_ptr() == source.data_ptr(),
          "CUDA copy assignment must share storage");
    Tensor& same = assigned;
    assigned = same;
    Tensor moved(std::move(copied));
    Check(copied.empty() && moved.data_ptr() == source.data_ptr(),
          "CUDA move constructor must transfer storage");
    Tensor moved_again;
    Check(&(moved_again = std::move(moved)) == &moved_again && moved.empty(),
          "CUDA move assignment must transfer storage");
    source = Tensor();
    assigned = Tensor();
    CheckValues<float>(moved_again, {7, 7, 3, 4});
}

void TestCloneAndCopy() {
    float input[] = {1, 2, 3, 4, 5, 6};
    Tensor source(input, {2, 3}, Dtype::kFloat32, Device::kCUDA);
    Tensor clone = source.clone();
    Check(clone.data_ptr() != source.data_ptr() && clone.shape() == source.shape() &&
          clone.dtype() == source.dtype() && clone.device() == source.device(),
          "CUDA clone must allocate independent storage and preserve metadata");
    CheckValues<float>(clone, {1, 2, 3, 4, 5, 6});
    clone.zero_();
    CheckValues<float>(source, {1, 2, 3, 4, 5, 6});

    Tensor first = source[0];
    Tensor second = source[1];
    Check(&second.copy_(first) == &second, "CUDA copy_ must return target");
    CheckValues<float>(source, {1, 2, 3, 1, 2, 3});
    second.copy_(second);
    Tensor alias = second;
    second.copy_(alias);
    Tensor view_clone = second.clone();
    source = Tensor();
    second = Tensor();
    CheckValues<float>(view_clone, {1, 2, 3});

    Tensor empty;
    Check(empty.clone().empty(), "Cloning an empty Tensor failed");
    Check(&empty.copy_(Tensor()) == &empty, "Empty copy_ must return target");
}

template <typename T>
void TestNativeFill(Dtype dtype) {
    T input[] = {1, 2, 3, 4};
    Tensor tensor(input, {4}, dtype, Device::kCUDA);
    Check(&tensor.fill_(7) == &tensor, "CUDA fill_ must return target");
    CheckValues<T>(tensor, {7, 7, 7, 7});
    Check(&tensor.zero_() == &tensor, "CUDA zero_ must return target");
    CheckValues<T>(tensor, {0, 0, 0, 0});
}

void TestFillAndZero() {
    TestNativeFill<float>(Dtype::kFloat32);
    TestNativeFill<std::int8_t>(Dtype::kInt8);
    TestNativeFill<std::uint8_t>(Dtype::kUInt8);
    TestNativeFill<std::int16_t>(Dtype::kInt16);
    TestNativeFill<std::int32_t>(Dtype::kInt32);
    TestNativeFill<std::int64_t>(Dtype::kInt64);

    const Dtype half_types[] = {Dtype::kFloat16, Dtype::kBFloat16};
    for (Dtype dtype : half_types) {
        std::uint16_t input[] = {0, 0, 0, 0};
        Tensor tensor(input, {2, 2}, dtype, Device::kCUDA);
        tensor.fill_(1.0f);
        const std::uint16_t one = dtype == Dtype::kFloat16 ? 0x3c00u : 0x3f80u;
        CheckValues<std::uint16_t>(tensor, {one, one, one, one});
        Tensor view = tensor[1];
        view.fill_(-1.5f);
        const std::uint16_t minus_one_half =
            dtype == Dtype::kFloat16 ? 0xbe00u : 0xbfc0u;
        CheckValues<std::uint16_t>(tensor, {one, one, minus_one_half, minus_one_half});
        view.zero_();
        CheckValues<std::uint16_t>(tensor, {one, one, 0, 0});
    }

    std::int64_t wide_input[] = {0, 0};
    Tensor wide(wide_input, {2}, Dtype::kInt64, Device::kCUDA);
    const std::int64_t large = (std::int64_t{1} << 60) + 3;
    wide.fill_(large);
    CheckValues<std::int64_t>(wide, {large, large});

    std::vector<std::int32_t> long_input(1025, 0);
    Tensor long_tensor(long_input.data(), {1025}, Dtype::kInt32, Device::kCUDA);
    long_tensor.fill_(42);
    CheckValues<std::int32_t>(long_tensor, std::vector<std::int32_t>(1025, 42));
}

void TestStridedZero() {
    std::int32_t input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor tensor(input, {3, 4}, Dtype::kInt32, Device::kCUDA);
    Tensor columns = tensor.narrow(1, 1, 2);
    Check(&columns.zero_() == &columns, "strided zero_ must return target");
    CheckValues<std::int32_t>(tensor, {1, 0, 0, 4, 5, 0, 0, 8, 9, 0, 0, 12});

    Tensor nested = tensor.narrow(0, 1, 2).narrow(1, 3, 1);
    nested.zero_();
    CheckValues<std::int32_t>(tensor, {1, 0, 0, 4, 5, 0, 0, 0, 9, 0, 0, 0});

    std::uint16_t half_input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor half(half_input, {2, 2, 3}, Dtype::kFloat16, Device::kCUDA);
    Tensor half_view = half.narrow(2, 1, 1);
    half_view.zero_();
    CheckValues<std::uint16_t>(half, {1, 0, 3, 4, 0, 6, 7, 0, 9, 10, 0, 12});

    float scalar_input = 3.5f;
    Tensor scalar(&scalar_input, {}, Dtype::kFloat32, Device::kCUDA);
    scalar.zero_();
    CheckValues<float>(scalar, {0.0f});

    std::vector<std::int32_t> large_input(3 * 259, 7);
    Tensor large(large_input.data(), {3, 259}, Dtype::kInt32, Device::kCUDA);
    Tensor large_view = large.narrow(1, 1, 257);
    large_view.zero_();
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 1; col < 258; ++col) {
            large_input[row * 259 + col] = 0;
        }
    }
    CheckValues<std::int32_t>(large, large_input);
}

void TestStridedOperations() {
    float values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor cpu(values, {3, 4}, Dtype::kFloat32);
    Tensor gpu = cpu.to(Device::kCUDA);
    Tensor src = gpu.narrow(1, 1, 2);
    Tensor cpu_view = cpu.narrow(1, 1, 2);
    Tensor cloned = src.clone();
    CheckValues<float>(cloned, {2, 3, 6, 7, 10, 11});
    Tensor host = src.to(Device::kCPU);
    Check(std::memcmp(host.data_ptr(), std::vector<float>{2, 3, 6, 7, 10, 11}.data(), host.nbytes()) == 0,
          "non-contiguous CUDA to CPU failed");
    Tensor from_cpu = cpu_view.to(Device::kCUDA);
    CheckValues<float>(from_cpu, {2, 3, 6, 7, 10, 11});
    Check(Printed(src) == "tensor([[2, 3],\n        [6, 7],\n        [10, 11]])\n",
          "non-contiguous CUDA print failed");

    Tensor dst({3, 4}, Dtype::kFloat32, Device::kCUDA);
    dst.fill_(9);
    Tensor dst_view = dst.narrow(1, 0, 2);
    dst_view.copy_(src);
    CheckValues<float>(dst, {2, 3, 9, 9, 6, 7, 9, 9, 10, 11, 9, 9});
    Tensor packed({3, 2}, Dtype::kFloat32, Device::kCUDA);
    packed.copy_(src);
    CheckValues<float>(packed, {2, 3, 6, 7, 10, 11});
    Tensor other_columns = dst.narrow(1, 2, 2);
    other_columns.copy_(packed);
    CheckValues<float>(dst, {2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11});
    src.fill_(13);
    CheckValues<float>(gpu, {1, 13, 13, 4, 5, 13, 13, 8, 9, 13, 13, 12});
    CheckValues<float>(cloned, {2, 3, 6, 7, 10, 11});

    std::uint16_t bits[] = {0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600};
    Tensor half(bits, {2, 3}, Dtype::kFloat16, Device::kCUDA);
    Tensor half_view = half.narrow(1, 1, 1);
    Tensor half_copy = half_view.clone();
    CheckValues<std::uint16_t>(half_copy, {0x4000, 0x4500});
    half_view.fill_(1.0f);
    CheckValues<std::uint16_t>(half, {0x3c00, 0x3c00, 0x4200, 0x4400, 0x3c00, 0x4600});

    std::int64_t wide_bits[] = {1, 2, 3, 4, 5, 6};
    Tensor wide(wide_bits, {2, 3}, Dtype::kInt64, Device::kCUDA);
    Tensor wide_copy = wide.narrow(1, 1, 1).clone();
    CheckValues<std::int64_t>(wide_copy, {2, 5});
}

void TestPrint() {
    float input[] = {1, 2, 3, 4};
    Tensor tensor(input, {2, 2}, Dtype::kFloat32, Device::kCUDA);
    Check(Printed(tensor) == "tensor([[1, 2],\n        [3, 4]])\n",
          "CUDA matrix print is wrong");
    Check(Printed(tensor[1]) == "tensor([3, 4])\n", "CUDA view print is wrong");
    Check(Printed(tensor[1][0]) == "tensor(3)\n", "CUDA scalar print is wrong");
    std::uint16_t half = 0x3c00u;
    Tensor fp16(&half, {}, Dtype::kFloat16, Device::kCUDA);
    Check(Printed(fp16) == "tensor(1)\n", "CUDA FP16 print is wrong");
    half = 0x3f80u;
    Tensor bf16(&half, {}, Dtype::kBFloat16, Device::kCUDA);
    Check(Printed(bf16) == "tensor(1)\n", "CUDA BF16 print is wrong");
}

template <typename T>
void TestDeviceTransfer(Dtype dtype) {
    T input[] = {1, 2, 3, 4};
    const Tensor cpu(input, {2, 2}, dtype);
    Tensor gpu = cpu.to(Device::kCUDA);
    // 独立验证 CPU→CUDA，避免只看往返结果而掩盖传输方向上的错误。
    CheckValues<T>(gpu, {1, 2, 3, 4});
    Tensor host = gpu.to(Device::kCPU);
    const Tensor original_gpu(input, {2, 2}, dtype, Device::kCUDA);
    Tensor direct_host = original_gpu.to(Device::kCPU);
    Check(std::memcmp(direct_host.data_ptr(), input, sizeof(input)) == 0,
          "CUDA to CPU must work independently of CPU to CUDA");
    Check(cpu.device() == Device::kCPU && gpu.device() == Device::kCUDA &&
          host.device() == Device::kCPU, "to changed the source or lost target device");
    Check(gpu.shape() == cpu.shape() && host.shape() == cpu.shape() &&
          gpu.dtype() == dtype && host.dtype() == dtype, "to lost shape or dtype");
    Check(host.data_ptr() != cpu.data_ptr(), "Cross-device round-trip must own independent storage");
    Check(std::memcmp(host.data_ptr(), input, sizeof(input)) == 0, "to changed data bytes");

    Tensor gpu_view = cpu[1].to(Device::kCUDA);
    Tensor host_view = gpu[1].to(Device::kCPU);
    Check(gpu_view.shape() == std::vector<int>({2}) && host_view.shape() == gpu_view.shape(),
          "to lost view shape");
    CheckValues<T>(gpu_view, {3, 4});
    Check(std::memcmp(host_view.data_ptr(), input + 2, 2 * sizeof(T)) == 0,
          "to copied from the wrong view offset");

    Tensor same = gpu.to(Device::kCUDA);
    Check(same.data_ptr() == gpu.data_ptr(), "Same-device CUDA to must share storage");
    same.zero_();
    CheckValues<T>(gpu, {0, 0, 0, 0});
    Check(std::memcmp(host.data_ptr(), input, sizeof(input)) == 0 &&
          std::memcmp(cpu.data_ptr(), input, sizeof(input)) == 0,
          "GPU changes affected independent CPU storage");
}

void TestTo() {
    TestDeviceTransfer<float>(Dtype::kFloat32);
    TestDeviceTransfer<std::uint16_t>(Dtype::kFloat16);
    TestDeviceTransfer<std::uint16_t>(Dtype::kBFloat16);
    TestDeviceTransfer<std::int8_t>(Dtype::kInt8);
    TestDeviceTransfer<std::uint8_t>(Dtype::kUInt8);
    TestDeviceTransfer<std::int16_t>(Dtype::kInt16);
    TestDeviceTransfer<std::int32_t>(Dtype::kInt32);
    TestDeviceTransfer<std::int64_t>(Dtype::kInt64);

    float input[] = {1, 2, 3, 4};
    Tensor source(input, {2, 2}, Dtype::kFloat32);
    Tensor cpu_alias = source;
    Tensor cpu_view = source[1];
    source = source.to(Device::kCUDA);
    Check(source.device() == Device::kCUDA && cpu_alias.device() == Device::kCPU,
          "Assigning to changed a shared CPU Tensor's device");
    CheckValues<float>(source, {1, 2, 3, 4});
    source.fill_(9);
    Check(std::memcmp(cpu_alias.data_ptr(), input, sizeof(input)) == 0,
          "Assigning to destroyed or changed shared CPU storage");
    cpu_alias = Tensor();
    Check(static_cast<const float*>(cpu_view.data_ptr())[0] == 3,
          "CPU view must retain storage after its owner moves to CUDA");
    source = source.to(Device::kCPU);
    Check(source.device() == Device::kCPU && source.shape() == std::vector<int>({2, 2}),
          "Assigning to lost target device or shape");
    const auto* values = static_cast<const float*>(source.data_ptr());
    Check(values[0] == 9 && values[3] == 9, "Assigning to lost copied values");

    float scalar_value = 6;
    Tensor scalar(&scalar_value, {}, Dtype::kFloat32);
    scalar = scalar.to(Device::kCUDA);
    Check(scalar.shape().empty() && scalar.numel() == 1, "to lost scalar shape");
    CheckValues<float>(scalar, {6});
    scalar = scalar.to(Device::kCPU);
    Check(*static_cast<const float*>(scalar.data_ptr()) == 6, "Scalar round-trip failed");

    const std::uint16_t bits[] = {0x8000u, 0x7c00u, 0x7e01u, 0xffffu};
    for (Dtype dtype : {Dtype::kFloat16, Dtype::kBFloat16}) {
        Tensor encoded(bits, {4}, dtype);
        encoded = encoded.to(Device::kCUDA);
        encoded = encoded.to(Device::kCPU);
        Check(std::memcmp(encoded.data_ptr(), bits, sizeof(bits)) == 0,
              "to must preserve raw 16-bit encodings");
    }

    Tensor empty = Tensor().to(Device::kCUDA);
    Check(empty.empty() && empty.device() == Device::kCUDA && empty.shape().empty(),
          "Empty to must preserve empty state and set target device");
    empty = empty.to(Device::kCPU);
    Check(empty.empty() && empty.device() == Device::kCPU, "Empty return to CPU failed");
}

void TestToViewsAndLifetime() {
    float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor cpu(input, {2, 2, 3}, Dtype::kFloat32);
    Tensor gpu = cpu.to(Device::kCUDA);
    Tensor cpu_slice = gpu[1][1].to(Device::kCPU);
    Tensor gpu_slice = cpu[1][1].to(Device::kCUDA);
    Tensor shared_slice = gpu[1][1].to(Device::kCUDA);
    Check(shared_slice.data_ptr() == static_cast<float*>(gpu.data_ptr()) + 9,
          "Same-device CUDA to must preserve a nested view's data offset");
    Check(cpu_slice.shape() == std::vector<int>({3}) &&
          gpu_slice.shape() == cpu_slice.shape() && shared_slice.shape() == cpu_slice.shape(),
          "Nested view to lost shape");

    cpu = Tensor();
    gpu = Tensor();
    CheckValues<float>(gpu_slice, {10, 11, 12});
    CheckValues<float>(shared_slice, {10, 11, 12});
    Check(std::memcmp(cpu_slice.data_ptr(), input + 9, 3 * sizeof(float)) == 0,
          "Transferred slice must survive source destruction");

    // GPU→CPU 赋值后，已有 GPU 视图仍独立持有原来的 GPU 存储。
    const Tensor& read_only = shared_slice;
    Tensor gpu_alias = read_only.to(Device::kCUDA);
    shared_slice = shared_slice.to(Device::kCUDA);
    Check(shared_slice.data_ptr() == gpu_alias.data_ptr(),
          "CUDA same-device self assignment must retain storage");
    shared_slice = shared_slice.to(Device::kCPU);
    gpu_alias.fill_(42);
    CheckValues<float>(gpu_alias, {42, 42, 42});
    Check(shared_slice.device() == Device::kCPU &&
          std::memcmp(shared_slice.data_ptr(), input + 9, 3 * sizeof(float)) == 0,
          "GPU alias changes must not affect the migrated CPU result");

    const void* original_data = gpu_alias.data_ptr();
    bool threw = false;
    try {
        gpu_alias = gpu_alias.to(static_cast<Device>(255));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw && gpu_alias.data_ptr() == original_data &&
          gpu_alias.device() == Device::kCUDA && gpu_alias.dtype() == Dtype::kFloat32 &&
          gpu_alias.shape() == std::vector<int>({3}),
          "Failed CUDA to assignment must preserve source storage and metadata");
    CheckValues<float>(gpu_alias, {42, 42, 42});
}

void TestToEmptyMetadata() {
    std::int64_t value = 7;
    Tensor source(&value, {}, Dtype::kInt64, Device::kCUDA);
    Tensor owner(std::move(source));
    Check(source.empty(), "Move must leave an empty CUDA source");
    Tensor same = source.to(Device::kCUDA);
    Tensor cpu = source.to(Device::kCPU);
    Tensor back = cpu.to(Device::kCUDA);
    for (const Tensor* empty : {&same, &cpu, &back}) {
        Check(empty->empty() && empty->data_ptr() == nullptr && empty->numel() == 0 &&
              empty->nbytes() == 0 && empty->shape().empty() && empty->dtype() == Dtype::kInt64,
              "Empty to must preserve non-default dtype and avoid allocating scalar storage");
    }
    Check(same.device() == Device::kCUDA && cpu.device() == Device::kCPU &&
          back.device() == Device::kCUDA, "Empty to lost target device");
    CheckValues<std::int64_t>(owner, {7});
}

void TestToRawValues() {
    // 使用位模式比较，验证负零、无穷和 NaN payload，不触发浮点 NaN 比较。
    const std::uint32_t bits[] = {0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc01234u};
    Tensor cpu(bits, {4}, Dtype::kFloat32);
    Tensor gpu = cpu.to(Device::kCUDA);
    Check(Read<std::uint32_t>(gpu) == std::vector<std::uint32_t>(bits, bits + 4),
          "CPU to CUDA changed Float32 special-value bits");
    Tensor host = gpu.to(Device::kCPU);
    Check(std::memcmp(host.data_ptr(), bits, sizeof(bits)) == 0,
          "CUDA to CPU changed Float32 special-value bits");

    const std::int64_t wide[] = {
        std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max(),
        (std::int64_t{1} << 60) + 3, -(std::int64_t{1} << 60) - 5,
    };
    Tensor integers(wide, {4}, Dtype::kInt64);
    Tensor gpu_integers = integers.to(Device::kCUDA);
    CheckValues<std::int64_t>(gpu_integers, std::vector<std::int64_t>(wide, wide + 4));
    Tensor host_integers = gpu_integers.to(Device::kCPU);
    Check(std::memcmp(host_integers.data_ptr(), wide, sizeof(wide)) == 0,
          "to must preserve Int64 values without floating-point conversion");
}

void TestInvalidInputs() {
    float input[] = {1, 2, 3, 4};
    bool threw = false;
    try {
        Tensor invalid(nullptr, {1}, Dtype::kFloat32, Device::kCUDA);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "CUDA constructor must reject null source");
    threw = false;
    try {
        Tensor invalid(input, {-1}, Dtype::kFloat32, Device::kCUDA);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "CUDA constructor must reject negative dimensions");
    threw = false;
    try {
        Tensor invalid(input, {0}, Dtype::kFloat32, Device::kCUDA);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "CUDA constructor must reject zero-size input");

    Tensor target(input, {2, 2}, Dtype::kFloat32, Device::kCUDA);
    Tensor wrong_shape(input, {4}, Dtype::kFloat32, Device::kCUDA);
    threw = false;
    try {
        target.copy_(wrong_shape);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "CUDA copy_ must reject mismatched shape");
    std::int32_t integer[] = {1, 2, 3, 4};
    Tensor wrong_dtype(integer, {2, 2}, Dtype::kInt32, Device::kCUDA);
    threw = false;
    try {
        target.copy_(wrong_dtype);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "CUDA copy_ must reject mismatched dtype");
    Tensor cpu(input, {2, 2}, Dtype::kFloat32);
    threw = false;
    try {
        target.copy_(cpu);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "CUDA copy_ must reject cross-device copy");
    threw = false;
    try {
        target[2];
    } catch (const std::out_of_range&) {
        threw = true;
    }
    Check(threw, "CUDA view must reject out-of-range index");
    CheckValues<float>(target, {1, 2, 3, 4});
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && device_count == 0)) {
            std::cout << "Skipping CUDA Tensor tests: no usable GPU/driver\n";
            return 77;
        }
        CheckCuda(status);
        TestMemory();
        TestConstructionAndViews();
        TestOwnership();
        TestCloneAndCopy();
        TestFillAndZero();
        TestStridedZero();
        TestStridedOperations();
        TestPrint();
        TestTo();
        TestToViewsAndLifetime();
        TestToEmptyMetadata();
        TestToRawValues();
        TestInvalidInputs();
        CheckCuda(cudaDeviceSynchronize());
        std::cout << "Tensor CUDA tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
