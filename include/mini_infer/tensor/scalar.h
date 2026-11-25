#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace mtensor {

// 标量保留数值类别，避免大整数经过浮点数时丢失精度。
class Scalar {
public:
    template <typename T>
    Scalar(T value) {
        static_assert(std::is_arithmetic_v<T>, "Scalar requires a numeric type");

        if constexpr (std::is_integral_v<T>) {
            static_assert(sizeof(T) <= sizeof(std::uint64_t), "Scalar integer is too wide");
            if constexpr (std::is_signed_v<T>) {
                kind_ = Kind::kInt;
                value_.signed_value = static_cast<std::int64_t>(value);
            } else {
                kind_ = Kind::kUInt;
                value_.unsigned_value = static_cast<std::uint64_t>(value);
            }
        } else if constexpr (std::is_floating_point_v<T>) {
            static_assert(sizeof(T) <= sizeof(double), "Scalar floating-point value is too wide");
            kind_ = Kind::kFloat;
            value_.float_value = static_cast<double>(value);
        }
    }

    bool is_floating_point() const { return kind_ == Kind::kFloat; }
    bool is_unsigned_integer() const { return kind_ == Kind::kUInt; }

    template <typename T>
    T to() const {
        static_assert(std::is_arithmetic_v<T>, "Scalar::to requires an arithmetic type");
        switch (kind_) {
            case Kind::kInt:
                return static_cast<T>(value_.signed_value);
            case Kind::kUInt:
                return static_cast<T>(value_.unsigned_value);
            case Kind::kFloat:
                if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                    // 浮点数转整数超出表示范围或为 NaN/Inf 时不能直接 static_cast。
                    const long double truncated = std::trunc(static_cast<long double>(value_.float_value));
                    const long double upper = std::ldexp(1.0L, std::numeric_limits<T>::digits);
                    const long double lower = std::is_signed_v<T> ? -upper : 0.0L;
                    if (!std::isfinite(truncated) || truncated < lower || truncated >= upper) {
                        throw std::out_of_range("Scalar: floating-point value is out of integer range");
                    }
                }
                return static_cast<T>(value_.float_value);
        }
        throw std::logic_error("Scalar: invalid kind");
    }

private:
    enum class Kind { kInt, kUInt, kFloat };

    union Value {
        std::int64_t signed_value;
        std::uint64_t unsigned_value;
        double float_value;
    } value_{};
    Kind kind_;
};

}  // namespace mtensor
