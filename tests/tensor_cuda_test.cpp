#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
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
    Check(tensor.size() == std::vector<int>({2, 2}) && tensor.size(1) == 2 &&
          tensor.dim() == 2 && tensor.numel() == 4 &&
          tensor.element_size() == sizeof(T) && tensor.nbytes() == sizeof(input),
          "CUDA Tensor shape or byte count is wrong");
    CheckValues<T>(tensor, {1, 2, 3, 4});
    input[0] = 99;
    CheckValues<T>(tensor, {1, 2, 3, 4});

    Tensor row = tensor[1];
    Check(row.size() == std::vector<int>({2}) && row.device() == Device::kCUDA &&
          row.dtype() == dtype && row.data_ptr() == static_cast<T*>(tensor.data_ptr()) + 2,
          "CUDA view metadata or offset is wrong");
    CheckValues<T>(row, {3, 4});
    Tensor scalar = row[1];
    Check(scalar.size().empty() && scalar.dim() == 0 && scalar.numel() == 1,
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
    Check(clone.data_ptr() != source.data_ptr() && clone.size() == source.size() &&
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
        CheckCuda(cudaGetDeviceCount(&device_count));
        Check(device_count > 0, "CUDA Tensor tests require a GPU");
        TestMemory();
        TestConstructionAndViews();
        TestOwnership();
        TestCloneAndCopy();
        TestFillAndZero();
        TestPrint();
        TestInvalidInputs();
        CheckCuda(cudaDeviceSynchronize());
        std::cout << "Tensor CUDA tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
