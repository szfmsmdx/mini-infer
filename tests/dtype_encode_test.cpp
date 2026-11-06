#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "tensor/tensor.h"

namespace {

void Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float FloatFromBits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct EncodeCase {
    std::uint32_t input;
    std::uint16_t expected;
};

void TestBFloat16() {
    // 直接指定 FP32 位模式，精确覆盖舍入中点及其相邻值。
    const EncodeCase cases[] = {
        {0x00000000u, 0x0000u}, {0x80000000u, 0x8000u},
        {0x3f800000u, 0x3f80u}, {0x40c00000u, 0x40c0u},
        {0xc0c00000u, 0xc0c0u},
        {0x3f807fffu, 0x3f80u}, {0x3f808000u, 0x3f80u},
        {0x3f808001u, 0x3f81u}, {0x3f818000u, 0x3f82u},
        {0xbf808001u, 0xbf81u},
        {0x00008000u, 0x0000u}, {0x00008001u, 0x0001u},
        {0x00018000u, 0x0002u}, {0x007fffffu, 0x0080u},
        {0x7f7fffffu, 0x7f80u}, {0x7f800000u, 0x7f80u},
        {0xff800000u, 0xff80u}, {0x7f800001u, 0x7fc0u},
        {0xff800001u, 0xffc0u},
    };
    for (const EncodeCase& entry : cases) {
        Check(mtensor::EncodeBFloat16(FloatFromBits(entry.input)) == entry.expected,
              "BF16 encoding or rounding failed");
    }
}

void TestFloat16() {
    const EncodeCase cases[] = {
        {0x00000000u, 0x0000u}, {0x80000000u, 0x8000u},
        {0x3f800000u, 0x3c00u}, {0x40c00000u, 0x4600u},
        {0xc0c00000u, 0xc600u},
        {0x3f800fffu, 0x3c00u}, {0x3f801000u, 0x3c00u},
        {0x3f801001u, 0x3c01u}, {0x3f803000u, 0x3c02u},
        {0xbf801001u, 0xbc01u},
        {0x477fe000u, 0x7bffu}, {0x477fefffu, 0x7bffu},
        {0x477ff000u, 0x7c00u}, {0x7f7fffffu, 0x7c00u},
        {0x38800000u, 0x0400u}, {0x387fc000u, 0x03ffu},
        {0x387fe000u, 0x0400u}, {0x33800000u, 0x0001u},
        {0x33000000u, 0x0000u}, {0x33000001u, 0x0001u},
        {0xb3800000u, 0x8001u}, {0xb3000000u, 0x8000u},
        {0x00000001u, 0x0000u}, {0x7f800000u, 0x7c00u},
        {0xff800000u, 0xfc00u}, {0x7f800001u, 0x7e00u},
        {0xff800001u, 0xfe00u},
    };
    for (const EncodeCase& entry : cases) {
        Check(mtensor::EncodeFloat16(FloatFromBits(entry.input)) == entry.expected,
              "FP16 encoding or rounding failed");
    }
}

void TestRepresentableValues() {
    // 独立按数值公式生成所有可表示的 FP16 值，检查编码能否还原原位模式。
    // NaN 的 payload 可以规范化，因此不要求逐位还原 NaN。
    for (std::uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const int exponent = static_cast<int>((bits >> 10) & 0x1fu);
        const int fraction = static_cast<int>(bits & 0x03ffu);
        if (exponent == 31 && fraction != 0) {
            continue;
        }
        float value;
        if (exponent == 0) {
            value = std::ldexp(static_cast<float>(fraction), -24);
        } else if (exponent == 31) {
            value = std::numeric_limits<float>::infinity();
        } else {
            value = std::ldexp(1.0f + fraction / 1024.0f, exponent - 15);
        }
        if ((bits & 0x8000u) != 0) {
            value = -value;
        }
        Check(mtensor::EncodeFloat16(value) == bits,
              "FP16 representable value did not round-trip");
    }

    for (std::uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        if ((bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0) {
            continue;
        }
        Check(mtensor::EncodeBFloat16(FloatFromBits(bits << 16)) == bits,
              "BF16 representable value did not round-trip");
    }
}

void CheckFilled(mtensor::Tensor& tensor, std::uint16_t expected) {
    const std::uint16_t* data = static_cast<const std::uint16_t*>(tensor.data_ptr());
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        Check(data[i] == expected, "Tensor::fill_ stored incorrect 16-bit data");
    }
}

void TestTensorFill() {
    const mtensor::Dtype types[] = {
        mtensor::Dtype::kFloat16, mtensor::Dtype::kBFloat16,
    };
    for (mtensor::Dtype dtype : types) {
        const std::uint16_t initial[] = {0, 0, 0, 0, 0, 0};
        mtensor::Tensor tensor(initial, {2, 3}, dtype);
        Check(&tensor.fill_(6.0f) == &tensor, "fill_ must return the same Tensor");
        const std::uint16_t six = dtype == mtensor::Dtype::kFloat16 ? 0x4600u : 0x40c0u;
        CheckFilled(tensor, six);

        mtensor::Tensor scalar = tensor[1][2];
        scalar.fill_(-1.5f);
        const std::uint16_t minus_one_half =
            dtype == mtensor::Dtype::kFloat16 ? 0xbe00u : 0xbfc0u;
        Check(scalar.numel() == 1, "Scalar view must contain one element");
        CheckFilled(scalar, minus_one_half);
        const std::uint16_t* data = static_cast<const std::uint16_t*>(tensor.data_ptr());
        Check(data[5] == minus_one_half && data[0] == six,
              "Filling a scalar view must change only its source element");

        mtensor::Tensor copy = tensor.clone();
        copy.fill_(1);  // 整数输入也应先转换为浮点编码。
        CheckFilled(copy, dtype == mtensor::Dtype::kFloat16 ? 0x3c00u : 0x3f80u);
        Check(data[0] == six && data[5] == minus_one_half,
              "Filling a clone must not change the original Tensor");
    }

    float initial[] = {0, 0};
    mtensor::Tensor tensor(initial, {2}, mtensor::Dtype::kFloat32);
    tensor.fill_(1.5f);
    const float* data = static_cast<const float*>(tensor.data_ptr());
    Check(data[0] == 1.5f && data[1] == 1.5f, "Float32 fill regression");
}

}  // namespace

int main() {
    try {
        const mtensor::Float16 fp16(1.5f);
        const mtensor::BFloat16 bf16(1.5f);
        Check(fp16.bits == 0x3e00u && static_cast<float>(fp16) == 1.5f,
              "Float16 storage and conversion");
        Check(bf16.bits == 0x3fc0u && static_cast<float>(bf16) == 1.5f,
              "BFloat16 storage and conversion");
        Check(mtensor::dtype_info(mtensor::Dtype::kInt8).size() == 1 &&
              !mtensor::dtype_info(mtensor::Dtype::kInt8).is_quantized(),
              "Int8 dtype metadata");
        TestBFloat16();
        TestFloat16();
        TestRepresentableValues();
        TestTensorFill();
        std::cout << "dtype encoding and fill tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
