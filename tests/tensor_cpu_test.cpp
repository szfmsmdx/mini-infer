#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensor/tensor.h"

namespace {

using mtensor::Device;
using mtensor::Dtype;
using mtensor::Tensor;

// 不用 assert，避免 Release 构建关闭断言后漏掉测试。
void Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void CheckValues(Tensor& tensor, const std::vector<T>& expected) {
    Check(tensor.numel() == expected.size(), "Unexpected element count");
    const T* values = static_cast<const T*>(tensor.data_ptr());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        Check(values[i] == expected[i], "Unexpected Tensor element");
    }
}

void CheckIndexThrows(const Tensor& tensor, int index) {
    bool threw = false;
    try {
        tensor[index];
    } catch (const std::out_of_range&) {
        threw = true;
    }
    Check(threw, "Invalid index must throw std::out_of_range");
}

void CheckCopyThrows(Tensor& target, const Tensor& source) {
    bool threw = false;
    try {
        target.copy_(source);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "Incompatible copy_ must throw std::invalid_argument");
}

// 捕获 print() 的输出，同时在异常路径恢复 cout。
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

void TestEmptyTensor() {
    Tensor tensor;
    Check(tensor.empty(), "Default Tensor must be empty");
    Check(tensor.data_ptr() == nullptr, "Default data pointer must be null");
    Check(tensor.size().empty() && tensor.dim() == 0, "Default shape must be empty");
    Check(tensor.numel() == 0 && tensor.nbytes() == 0, "Empty Tensor has no elements");
    Check(tensor.dtype() == Dtype::kFloat32 && tensor.device() == Device::kCPU,
          "Unexpected default dtype or device");
    Check(&tensor.fill_(3) == &tensor, "Empty fill_ must return the same Tensor");
    Check(&tensor.zero_() == &tensor, "Empty zero_ must return the same Tensor");
    Tensor other;
    Check(&tensor.copy_(other) == &tensor, "Empty copy_ must return the target");
    Tensor cloned = tensor.clone();
    Check(cloned.empty() && cloned.numel() == 0, "Cloning an empty Tensor failed");
    Check(Printed(tensor) == "tensor([])\n", "Unexpected empty Tensor print");
    CheckIndexThrows(tensor, 0);
}

void TestConstruction() {
    float input[] = {1, 2, 3, 4, 5, 6};
    Tensor tensor(input, {2, 3}, Dtype::kFloat32);
    Check(tensor.size() == std::vector<int>({2, 3}), "Constructor lost shape");
    Check(tensor.dim() == 2 && tensor.size(0) == 2 && tensor.size(1) == 3,
          "Unexpected dimensions");
    Check(tensor.numel() == 6 && tensor.element_size() == sizeof(float) &&
          tensor.nbytes() == sizeof(input), "Unexpected Tensor size in memory");
    Check(!tensor.empty() && tensor.device() == Device::kCPU,
          "CPU Tensor must own data");
    Check(tensor.data_ptr() != input, "Constructor must copy external data");
    input[0] = 99;
    CheckValues<float>(tensor, {1, 2, 3, 4, 5, 6});
    tensor.fill_(7);
    Check(input[0] == 99 && input[1] == 2, "Tensor modified the source array");
}

template <typename T>
void TestStorageAndIndex(Dtype dtype) {
    T input[] = {1, 2, 3, 4};
    Tensor tensor(input, {2, 2}, dtype);
    Check(tensor.dtype() == dtype && tensor.element_size() == sizeof(T),
          "dtype storage size mismatch");
    Check(tensor.nbytes() == sizeof(input), "dtype byte count mismatch");
    Tensor row = tensor[1];
    Check(row.size() == std::vector<int>({2}), "Index must remove first dimension");
    Check(row.data_ptr() == static_cast<T*>(tensor.data_ptr()) + 2,
          "Index offset is incorrect for dtype");
    CheckValues<T>(row, {3, 4});
    Tensor scalar = row[0];
    Check(scalar.dim() == 0 && scalar.numel() == 1 && !scalar.empty(),
          "Scalar view must contain one element");
    Check(scalar.nbytes() == sizeof(T), "Scalar byte count mismatch");
    CheckValues<T>(scalar, {3});
    Tensor cloned = row.clone();
    Check(cloned.data_ptr() != row.data_ptr(), "View clone must allocate storage");
    CheckValues<T>(cloned, {3, 4});
}

void TestAllDtypeStorage() {
    TestStorageAndIndex<float>(Dtype::kFloat32);
    TestStorageAndIndex<std::int8_t>(Dtype::kInt8);
    TestStorageAndIndex<std::uint8_t>(Dtype::kUInt8);
    TestStorageAndIndex<std::int16_t>(Dtype::kInt16);
    TestStorageAndIndex<std::int32_t>(Dtype::kInt32);
    TestStorageAndIndex<std::int64_t>(Dtype::kInt64);
    // 这里验证原始编码的存储和偏移；数值 encode 由 dtype_encode_test 验证。
    TestStorageAndIndex<std::uint16_t>(Dtype::kFloat16);
    TestStorageAndIndex<std::uint16_t>(Dtype::kBFloat16);
}

void TestScalar() {
    float input = 6;
    Tensor scalar(&input, {}, Dtype::kFloat32);
    Check(scalar.dim() == 0 && scalar.size().empty(), "Scalar shape must be {}");
    Check(!scalar.empty() && scalar.numel() == 1, "Scalar is not an empty Tensor");
    Check(scalar.nbytes() == sizeof(float), "Scalar must occupy one element");
    Check(Printed(scalar) == "tensor(6)\n", "Unexpected scalar print");
    scalar.fill_(2.5f);
    CheckValues<float>(scalar, {2.5f});
    Check(input == 6, "Scalar constructor must copy external input");
    Tensor cloned = scalar.clone();
    cloned.zero_();
    CheckValues<float>(scalar, {2.5f});
    CheckValues<float>(cloned, {0});
    CheckIndexThrows(scalar, 0);
}

void TestSharedCopies() {
    float input[] = {1, 2, 3};
    Tensor source(input, {3}, Dtype::kFloat32);
    Tensor copied(source);
    Check(copied.data_ptr() == source.data_ptr(), "Copy constructor must share data");
    copied.fill_(4);
    CheckValues<float>(source, {4, 4, 4});

    std::int8_t old_input[] = {1};
    Tensor assigned(old_input, {1}, Dtype::kInt8);
    Check(&(assigned = source) == &assigned, "Copy assignment must return target");
    Check(assigned.data_ptr() == source.data_ptr() &&
          assigned.size() == source.size() && assigned.dtype() == source.dtype(),
          "Copy assignment must replace metadata and share storage");
    Tensor& same_object = assigned;
    assigned = same_object;
    CheckValues<float>(assigned, {4, 4, 4});

    source = Tensor();
    copied = Tensor();
    CheckValues<float>(assigned, {4, 4, 4});
    assigned.zero_();  // 原对象释放后，最后一个共享持有者仍能读写。
    CheckValues<float>(assigned, {0, 0, 0});
}

void TestMove() {
    float input[] = {1, 2, 3, 4};
    Tensor source(input, {2, 2}, Dtype::kFloat32);
    void* original_data = source.data_ptr();
    Tensor moved(std::move(source));
    Check(moved.data_ptr() == original_data, "Move constructor must transfer storage");
    CheckValues<float>(moved, {1, 2, 3, 4});
    Check(source.empty() && source.size().empty() && source.numel() == 0,
          "Move constructor must clear source state");

    float old_input[] = {9};
    Tensor assigned(old_input, {1}, Dtype::kFloat32);
    Check(&(assigned = std::move(moved)) == &assigned,
          "Move assignment must return target");
    Check(assigned.data_ptr() == original_data && assigned.dim() == 2,
          "Move assignment must transfer data and shape");
    Check(moved.empty() && moved.size().empty() && moved.numel() == 0,
          "Move assignment must clear source state");
    Tensor& same_object = assigned;
    assigned = std::move(same_object);
    CheckValues<float>(assigned, {1, 2, 3, 4});

    Tensor view = assigned[1];
    Tensor moved_view(std::move(view));
    Check(view.empty(), "Moving a view must clear source");
    Check(moved_view.data_ptr() == static_cast<float*>(original_data) + 2,
          "Moving a view must preserve its offset");
    assigned = Tensor();
    CheckValues<float>(moved_view, {3, 4});
    source = moved_view.clone();  // 移动后的对象可以重新赋值。
    CheckValues<float>(source, {3, 4});
}

void TestViewsAndLifetime() {
    Tensor row;
    Tensor scalar;
    {
        float input[] = {1, 2, 3, 4, 5, 6};
        Tensor source(input, {2, 3}, Dtype::kFloat32);
        row = source[1];
        scalar = row[2];
        scalar.fill_(-6);
        CheckValues<float>(source, {1, 2, 3, 4, 5, -6});
        Check(row.dtype() == source.dtype() && row.device() == source.device(),
              "View must preserve dtype and device");
    }
    // 原 Tensor 和外部数组均已离开作用域，视图仍持有底层内存。
    CheckValues<float>(row, {4, 5, -6});
    CheckValues<float>(scalar, {-6});
    scalar.fill_(8);
    CheckValues<float>(row, {4, 5, 8});
    row.zero_();
    CheckValues<float>(scalar, {0});
}

void TestClone() {
    float input[] = {1, 2, 3, 4, 5, 6};
    Tensor source(input, {2, 3}, Dtype::kFloat32);
    Tensor cloned = source.clone();
    Check(cloned.data_ptr() != source.data_ptr(), "Clone must have independent data");
    Check(cloned.size() == source.size() && cloned.dtype() == source.dtype() &&
          cloned.device() == source.device(), "Clone must preserve metadata");
    cloned.zero_();
    CheckValues<float>(source, {1, 2, 3, 4, 5, 6});
    CheckValues<float>(cloned, {0, 0, 0, 0, 0, 0});

    Tensor row = source[1];
    Tensor cloned_row = row.clone();
    CheckValues<float>(cloned_row, {4, 5, 6});
    source = Tensor();
    row = Tensor();
    CheckValues<float>(cloned_row, {4, 5, 6});
}

void TestCopyElements() {
    float input[] = {1, 2, 3, 4, 5, 6};
    Tensor source(input, {2, 3}, Dtype::kFloat32);
    Tensor row0 = source[0];
    Tensor row1 = source[1];
    void* original_target = row1.data_ptr();
    Check(&row1.copy_(row0) == &row1, "copy_ must return target");
    Check(row1.data_ptr() == original_target, "copy_ must retain target storage");
    CheckValues<float>(source, {1, 2, 3, 1, 2, 3});
    row0.fill_(9);
    CheckValues<float>(row1, {1, 2, 3});
    row1.copy_(row1);
    Tensor same_data = row1;
    row1.copy_(same_data);
    CheckValues<float>(row1, {1, 2, 3});

    Tensor scalar0 = row0[0];
    Tensor scalar1 = row1[1];
    scalar1.copy_(scalar0);
    CheckValues<float>(row1, {1, 9, 3});
    Tensor empty;
    CheckCopyThrows(scalar0, empty);
    CheckCopyThrows(empty, scalar0);
    CheckValues<float>(scalar0, {9});
    Check(empty.empty(), "Failed copy_ must not change empty target");
}

template <typename T>
void TestNativeFill(Dtype dtype) {
    T input[] = {1, 2, 3};
    Tensor tensor(input, {3}, dtype);
    Check(&tensor.fill_(7) == &tensor, "fill_ must return target");
    CheckValues<T>(tensor, {7, 7, 7});
    Check(&tensor.zero_() == &tensor, "zero_ must return target");
    CheckValues<T>(tensor, {0, 0, 0});
}

void TestFillAndZero() {
    TestNativeFill<float>(Dtype::kFloat32);
    TestNativeFill<std::int8_t>(Dtype::kInt8);
    TestNativeFill<std::uint8_t>(Dtype::kUInt8);
    TestNativeFill<std::int16_t>(Dtype::kInt16);
    TestNativeFill<std::int32_t>(Dtype::kInt32);
    TestNativeFill<std::int64_t>(Dtype::kInt64);

    std::int64_t input[] = {0, 0};
    Tensor wide(input, {2}, Dtype::kInt64);
    const std::int64_t large_value = (std::int64_t{1} << 60) + 3;
    wide.fill_(large_value);
    CheckValues<std::int64_t>(wide, {large_value, large_value});

    const Dtype types[] = {Dtype::kFloat16, Dtype::kBFloat16};
    for (Dtype dtype : types) {
        std::uint16_t bits[] = {0xffffu, 0xffffu};
        Tensor tensor(bits, {2}, dtype);
        tensor.zero_();
        CheckValues<std::uint16_t>(tensor, {0, 0});
    }
}

void TestErrors() {
    float input[] = {1, 2, 3, 4};
    const std::vector<std::vector<int>> invalid_shapes = {
        {-1}, {2, -1}, {-1, -1}, {0}, {2, 0},
    };
    for (const std::vector<int>& shape : invalid_shapes) {
        bool threw = false;
        try {
            Tensor invalid(input, shape, Dtype::kFloat32);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        Check(threw, "Negative or zero-size input shape must be rejected");
    }
    bool threw = false;
    try {
        Tensor invalid(nullptr, {1}, Dtype::kFloat32);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "Null source data must be rejected");

    Tensor tensor(input, {2, 2}, Dtype::kFloat32);
    CheckIndexThrows(tensor, -1);
    CheckIndexThrows(tensor, 2);
    threw = false;
    try {
        tensor.size(2);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    Check(threw, "Invalid size dimension must throw std::out_of_range");

    Tensor wrong_shape(input, {4}, Dtype::kFloat32);
    CheckCopyThrows(tensor, wrong_shape);
    std::int32_t integer_input[] = {1, 2, 3, 4};
    Tensor wrong_dtype(integer_input, {2, 2}, Dtype::kInt32);
    CheckCopyThrows(tensor, wrong_dtype);
    CheckValues<float>(tensor, {1, 2, 3, 4});
}

void CheckPrinted16(Dtype dtype, std::uint16_t bits, double expected) {
    Tensor scalar(&bits, {}, dtype);
    // 用 double 解析，避免 BF16 很小的数在 stof 中触发下溢错误。
    const double actual = std::stod(Printed(scalar).substr(7));
    if (std::isnan(expected)) {
        Check(std::isnan(actual), "Decode must preserve NaN");
    } else if (std::isinf(expected)) {
        Check(actual == expected, "Decode must preserve signed infinity");
    } else if (expected == 0) {
        Check(actual == 0 && std::signbit(actual) == std::signbit(expected),
              "Decode must preserve signed zero");
    } else {
        // print 使用默认的六位有效数字，因此按相对误差检查输出数值。
        Check(std::abs(actual - expected) <= std::abs(expected) * 0.00001,
              "Decoded printed value is incorrect");
    }
}

void TestPrintAndDecode() {
    float input[] = {1, 2, 3, 4, 5, 6};
    Tensor tensor(input, {2, 3}, Dtype::kFloat32);
    Check(Printed(tensor) == "tensor([[1, 2, 3],\n        [4, 5, 6]])\n",
          "Unexpected nested Tensor print");
    Check(Printed(tensor[1]) == "tensor([4, 5, 6])\n", "View print offset is wrong");
    Check(Printed(tensor[1][2]) == "tensor(6)\n", "Unexpected scalar view print");

    std::int8_t signed_input[] = {-3, 7};
    Tensor signed_tensor(signed_input, {2}, Dtype::kInt8);
    Check(Printed(signed_tensor) == "tensor([-3, 7])\n", "int8 must print numbers");
    std::uint8_t unsigned_input[] = {0, 255};
    Tensor unsigned_tensor(unsigned_input, {2}, Dtype::kUInt8);
    Check(Printed(unsigned_tensor) == "tensor([0, 255])\n", "uint8 must print numbers");

    CheckPrinted16(Dtype::kFloat16, 0x4600u, 6);
    CheckPrinted16(Dtype::kFloat16, 0xc000u, -2);
    CheckPrinted16(Dtype::kFloat16, 0x0001u, std::ldexp(1.0, -24));
    CheckPrinted16(Dtype::kFloat16, 0x0400u, std::ldexp(1.0, -14));
    CheckPrinted16(Dtype::kFloat16, 0x7bffu, 65504);
    CheckPrinted16(Dtype::kFloat16, 0x0000u, 0.0);
    CheckPrinted16(Dtype::kFloat16, 0x8000u, -0.0);
    CheckPrinted16(Dtype::kFloat16, 0x7c00u, std::numeric_limits<double>::infinity());
    CheckPrinted16(Dtype::kFloat16, 0xfc00u, -std::numeric_limits<double>::infinity());
    CheckPrinted16(Dtype::kFloat16, 0x7e00u, std::numeric_limits<double>::quiet_NaN());

    CheckPrinted16(Dtype::kBFloat16, 0x40c0u, 6);
    CheckPrinted16(Dtype::kBFloat16, 0xc000u, -2);
    CheckPrinted16(Dtype::kBFloat16, 0x0001u, std::ldexp(1.0, -133));
    CheckPrinted16(Dtype::kBFloat16, 0x7f7fu, std::ldexp(255.0, 120));
    CheckPrinted16(Dtype::kBFloat16, 0x0000u, 0.0);
    CheckPrinted16(Dtype::kBFloat16, 0x8000u, -0.0);
    CheckPrinted16(Dtype::kBFloat16, 0x7f80u, std::numeric_limits<double>::infinity());
    CheckPrinted16(Dtype::kBFloat16, 0xff80u, -std::numeric_limits<double>::infinity());
    CheckPrinted16(Dtype::kBFloat16, 0x7fc0u, std::numeric_limits<double>::quiet_NaN());
}

}  // namespace

int main() {
    try {
        TestEmptyTensor();
        TestConstruction();
        TestAllDtypeStorage();
        TestScalar();
        TestSharedCopies();
        TestMove();
        TestViewsAndLifetime();
        TestClone();
        TestCopyElements();
        TestFillAndZero();
        TestErrors();
        TestPrintAndDecode();
        std::cout << "Tensor CPU tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
