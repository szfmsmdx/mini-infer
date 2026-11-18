#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mtensor{
    enum class Dtype : std::uint8_t{
        kFloat32,
        kFloat16,
        kBFloat16,
        kInt8,
        kUInt8,
        kInt16,
        kInt32,
        kInt64
    };

    class DtypeInfo {
    public:
        constexpr DtypeInfo(const char* name, std::size_t size, bool is_quantized)
            : name_(name), size_(size), is_quantized_(is_quantized) {}

        constexpr const char* name() const { return name_; }
        constexpr std::size_t size() const { return size_; }
        constexpr bool is_quantized() const { return is_quantized_; }

    private:
        const char* name_;
        std::size_t size_;
        bool is_quantized_;
    };

    constexpr DtypeInfo dtype_info(Dtype t) {
        switch (t) {
            case Dtype::kFloat32: return {"float32", 4, false};
            case Dtype::kFloat16: return {"float16", 2, false};
            case Dtype::kBFloat16: return {"bfloat16", 2, false};
            case Dtype::kInt8: return {"int8", 1, false};
            case Dtype::kUInt8: return {"uint8", 1, false};
            case Dtype::kInt16: return {"int16", 2, false};
            case Dtype::kInt32: return {"int32", 4, false};
            case Dtype::kInt64: return {"int64", 8, false};
        }
        return {"unknown", 0, false};
    }

    constexpr std::size_t dtype_size(Dtype t) { return dtype_info(t).size(); }
    constexpr const char* dtype_name(Dtype t) { return dtype_info(t).name(); }

    // 输入为 IEEE 754 的 32 位 float，输出是 BF16 的原始 16 位编码。
    // 舍入规则：取最近的值；恰好在中间时，让保留部分的最低位为 0。
    inline std::uint16_t EncodeBFloat16(float value) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));

        const std::uint32_t exponent = bits & 0x7f800000u;
        const std::uint32_t fraction = bits & 0x007fffffu;
        if (exponent == 0x7f800000u && fraction != 0) {
            // 保留符号和高位 NaN 信息，并设置 quiet NaN 位，避免截断成无穷。
            return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
        }

        // [ retained：高 16 位 ][ discarded：低 16 位 ]
        std::uint32_t retained = bits >> 16;
        const std::uint32_t discarded = bits & 0xffffu;
        // 低 16 位大于一半时进位；恰好一半时，仅当保留部分为奇数才进位。
        if (discarded > 0x8000u ||
            (discarded == 0x8000u && (retained & 1u) != 0)) {
            ++retained;
        }
        return static_cast<std::uint16_t>(retained);
    }

    // 输入为 IEEE 754 的 32 位 float，输出是 FP16 的原始 16 位编码。
    // 同样采用最近值、恰好居中时取偶数的舍入规则。
    inline std::uint16_t EncodeFloat16(float value) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));

        const std::uint32_t sign = (bits >> 16) & 0x8000u;  // fp16 的符号位
        const int exponent = static_cast<int>((bits >> 23) & 0xffu);
        const std::uint32_t fraction = bits & 0x007fffffu;

        if (exponent == 0xff) {
            // 指数全为 1：小数为 0 表示无穷，否则输出 quiet NaN。
            return static_cast<std::uint16_t>(
                sign | (fraction == 0 ? 0x7c00u : 0x7e00u));
        }

        // 普通数的指数偏置从 FP32 的 127 转换为 FP16 的 15。
        int half_exponent = exponent - 127 + 15;
        if (half_exponent >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }

        if (half_exponent <= 0) {
            if (half_exponent < -10) {
                // 比最小 FP16 非正规数的一半还小，舍入到带符号的零。
                return static_cast<std::uint16_t>(sign);
            }

            // FP16 非正规数不保存隐含的 1，因此把 FP32 的隐含位一起右移。
            const std::uint32_t significand = 0x00800000u | fraction;
            // 普通转换丢弃 13 位，再额外右移 1 - half_exponent 位。
            const int shift = 14 - half_exponent;
            std::uint32_t retained = significand >> shift;
            const std::uint32_t discarded = significand & ((1u << shift) - 1u);
            const std::uint32_t halfway = 1u << (shift - 1);
            if (discarded > halfway ||
                (discarded == halfway && (retained & 1u) != 0)) {
                ++retained;
            }
            // 进位到 0x0400 时，恰好成为最小的正规 FP16 数。
            return static_cast<std::uint16_t>(sign | retained);
        }

        // 普通数将小数从 23 位缩减到 10 位，丢弃低 13 位。
        std::uint32_t retained = fraction >> 13;
        const std::uint32_t discarded = fraction & 0x1fffu;
        if (discarded > 0x1000u ||
            (discarded == 0x1000u && (retained & 1u) != 0)) {
            ++retained;
        }
        if (retained == 0x0400u) {
            // 小数进位：1.111... 舍入为 10.000...，需要增加指数。
            retained = 0;
            ++half_exponent;
        }
        // half_exponent 进位到 31 时，小数已清零，拼出的编码就是无穷。
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(half_exponent) << 10) | retained);
    }

inline float DecodeFloat16(const void* element) {
    std::uint16_t bits;
    std::memcpy(&bits, element, sizeof(bits));

    const bool negative = (bits & 0x8000u) != 0;
    const int exponent = (bits >> 10) & 0x1f;
    const int fraction = bits & 0x03ff;

    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(fraction), -24);
    } else if (exponent == 0x1f) {
        value = fraction == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else {
        // 普通 Float16: (1 + fraction / 1024) * 2^(exponent - 15)。
        // 改写为 (1024 + fraction) * 2^(exponent - 25)，其中 25 = 10 + 15。
        value = std::ldexp(static_cast<float>(0x400 + fraction), exponent - 25);
    }
    return negative ? -value : value;
}

inline float DecodeBFloat16(const void* element) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::uint16_t bits;
    std::memcpy(&bits, element, sizeof(bits));
    const std::uint32_t float_bits = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &float_bits, sizeof(value));    // 只取前 16bit
    return value;
}

struct Float16 {
    std::uint16_t bits;

    explicit Float16(float value) : bits(EncodeFloat16(value)) {}
    static Float16 from_bits(std::uint16_t value) {
        Float16 result(0.0f);
        result.bits = value;
        return result;
    }
    explicit operator float() const { return DecodeFloat16(&bits); }
};

struct BFloat16 {
    std::uint16_t bits;

    explicit BFloat16(float value) : bits(EncodeBFloat16(value)) {}
    static BFloat16 from_bits(std::uint16_t value) {
        BFloat16 result(0.0f);
        result.bits = value;
        return result;
    }
    explicit operator float() const { return DecodeBFloat16(&bits); }
};

static_assert(sizeof(Float16) == sizeof(std::uint16_t));
static_assert(sizeof(BFloat16) == sizeof(std::uint16_t));

    enum class Device : std::uint8_t{
        kCPU,
        kCUDA
    };
}
