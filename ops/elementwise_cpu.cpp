#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "ops/elementwise.h"

namespace mtensor {
namespace {

// 整数按固定位宽回绕。先转无符号计算，避免有符号加法溢出。
template <typename T>
T AddWrap(T a, T b) {
    using U = std::make_unsigned_t<T>;
    const U sum = static_cast<U>(a) + static_cast<U>(b);

    T result;
    std::memcpy(&result, &sum, sizeof(result));
    return result;
}

template <typename T>
T SubWrap(T a, T b) {
    using U = std::make_unsigned_t<T>;
    const U diff = static_cast<U>(a) - static_cast<U>(b);

    T result;
    std::memcpy(&result, &diff, sizeof(result));
    return result;
}

template <typename T>
T MulWrap(T a, T b) {
    using U = std::make_unsigned_t<T>;
    U product;
    if constexpr (sizeof(U) < sizeof(std::uint64_t)) {
        // 8/16 位乘法会先提升为 int，先转成 uint64_t 避免中间结果溢出。
        product = static_cast<U>(
            static_cast<std::uint64_t>(static_cast<U>(a)) *
            static_cast<std::uint64_t>(static_cast<U>(b)));
    } else {
        // uint64_t 乘法本身按模 2^64 回绕。
        product = static_cast<U>(a) * static_cast<U>(b);
    }

    T result;
    std::memcpy(&result, &product, sizeof(result));
    return result;
}

// 整数除法截断到零；除零返回 0，最小值除以 -1 回绕为最小值。
template <typename T>
T DivValue(T a, T b) {
    if constexpr (std::is_integral<T>::value) {
        if (b == T{0}) {
            return T{0};
        }
        if constexpr (std::is_signed<T>::value) {
            if (a == std::numeric_limits<T>::min() && b == T{-1}) {
                return std::numeric_limits<T>::min();
            }
        }
        return static_cast<T>(a / b);
    } else {
        return a / b;
    }
}

template <typename T>
T NegValue(T value) {
    if constexpr (std::is_integral<T>::value) {
        using U = std::make_unsigned_t<T>;
        const U result = U{0} - static_cast<U>(value);

        T casted;
        std::memcpy(&casted, &result, sizeof(casted));
        return casted;
    } else {
        return -value;
    }
}

// FP16/BF16 共用 16 位编码：解码为 float 计算，再编码写回。
template <typename Half, typename Operation>
void HalfBinaryCPU(const void* a, const void* b, void* output, std::size_t count) {
    const auto* a_bytes = static_cast<const std::byte*>(a);
    const auto* b_bytes = static_cast<const std::byte*>(b);
    auto* output_bytes = static_cast<std::byte*>(output);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t a_bits, b_bits;
        const std::size_t offset = i * sizeof(std::uint16_t);
        std::memcpy(&a_bits, a_bytes + offset, sizeof(a_bits));
        std::memcpy(&b_bits, b_bytes + offset, sizeof(b_bits));
        const float a_value = static_cast<float>(Half::from_bits(a_bits));
        const float b_value = static_cast<float>(Half::from_bits(b_bits));
        const Half result(Operation{}(a_value, b_value));
        std::memcpy(output_bytes + offset, &result.bits, sizeof(result.bits));
    }
}

// FP16 和 BF16 的符号位都在第 15 位，取负直接翻转该位。
void NegHalfCPU(const void* a, void* output, std::size_t count) {
    const auto* input = static_cast<const std::uint16_t*>(a);
    auto* output_ptr = static_cast<std::uint16_t*>(output);

    for (std::size_t i = 0; i < count; ++i) {
        output_ptr[i] = static_cast<std::uint16_t>(input[i] ^ 0x8000u);
    }
}

template <typename T>
void AddTypedCPU(const T* a, const T* b, T* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if constexpr (std::is_integral<T>::value) {
            output[i] = AddWrap(a[i], b[i]);
        } else {
            output[i] = a[i] + b[i];
        }
    }
}

template <typename T>
void SubTypedCPU(const T* a, const T* b, T* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if constexpr (std::is_integral<T>::value) {
            output[i] = SubWrap(a[i], b[i]);
        } else {
            output[i] = a[i] - b[i];
        }
    }
}

template <typename T>
void MulTypedCPU(const T* a, const T* b, T* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if constexpr (std::is_integral<T>::value) {
            output[i] = MulWrap(a[i], b[i]);
        } else {
            output[i] = a[i] * b[i];
        }
    }
}

template <typename T>
void DivTypedCPU(const T* a, const T* b, T* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = DivValue(a[i], b[i]);
    }
}

template <typename T>
void NegTypedCPU(const T* a, T* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = NegValue(a[i]);
    }
}

}  // namespace

void AddCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    switch (dtype) {
        case Dtype::kFloat16: HalfBinaryCPU<Float16, std::plus<float>>(a, b, output, count); break;
        case Dtype::kBFloat16: HalfBinaryCPU<BFloat16, std::plus<float>>(a, b, output, count); break;
        case Dtype::kFloat32: AddTypedCPU(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(output), count); break;
        case Dtype::kInt8: AddTypedCPU(static_cast<const std::int8_t*>(a), static_cast<const std::int8_t*>(b), static_cast<std::int8_t*>(output), count); break;
        case Dtype::kUInt8: AddTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<const std::uint8_t*>(b), static_cast<std::uint8_t*>(output), count); break;
        case Dtype::kInt16: AddTypedCPU(static_cast<const std::int16_t*>(a), static_cast<const std::int16_t*>(b), static_cast<std::int16_t*>(output), count); break;
        case Dtype::kInt32: AddTypedCPU(static_cast<const std::int32_t*>(a), static_cast<const std::int32_t*>(b), static_cast<std::int32_t*>(output), count); break;
        case Dtype::kInt64: AddTypedCPU(static_cast<const std::int64_t*>(a), static_cast<const std::int64_t*>(b), static_cast<std::int64_t*>(output), count); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void SubCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    switch (dtype) {
        case Dtype::kFloat16: HalfBinaryCPU<Float16, std::minus<float>>(a, b, output, count); break;
        case Dtype::kBFloat16: HalfBinaryCPU<BFloat16, std::minus<float>>(a, b, output, count); break;
        case Dtype::kFloat32: SubTypedCPU(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(output), count); break;
        case Dtype::kInt8: SubTypedCPU(static_cast<const std::int8_t*>(a), static_cast<const std::int8_t*>(b), static_cast<std::int8_t*>(output), count); break;
        case Dtype::kUInt8: SubTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<const std::uint8_t*>(b), static_cast<std::uint8_t*>(output), count); break;
        case Dtype::kInt16: SubTypedCPU(static_cast<const std::int16_t*>(a), static_cast<const std::int16_t*>(b), static_cast<std::int16_t*>(output), count); break;
        case Dtype::kInt32: SubTypedCPU(static_cast<const std::int32_t*>(a), static_cast<const std::int32_t*>(b), static_cast<std::int32_t*>(output), count); break;
        case Dtype::kInt64: SubTypedCPU(static_cast<const std::int64_t*>(a), static_cast<const std::int64_t*>(b), static_cast<std::int64_t*>(output), count); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void MulCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    switch (dtype) {
        case Dtype::kFloat16: HalfBinaryCPU<Float16, std::multiplies<float>>(a, b, output, count); break;
        case Dtype::kBFloat16: HalfBinaryCPU<BFloat16, std::multiplies<float>>(a, b, output, count); break;
        case Dtype::kFloat32: MulTypedCPU(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(output), count); break;
        case Dtype::kInt8: MulTypedCPU(static_cast<const std::int8_t*>(a), static_cast<const std::int8_t*>(b), static_cast<std::int8_t*>(output), count); break;
        case Dtype::kUInt8: MulTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<const std::uint8_t*>(b), static_cast<std::uint8_t*>(output), count); break;
        case Dtype::kInt16: MulTypedCPU(static_cast<const std::int16_t*>(a), static_cast<const std::int16_t*>(b), static_cast<std::int16_t*>(output), count); break;
        case Dtype::kInt32: MulTypedCPU(static_cast<const std::int32_t*>(a), static_cast<const std::int32_t*>(b), static_cast<std::int32_t*>(output), count); break;
        case Dtype::kInt64: MulTypedCPU(static_cast<const std::int64_t*>(a), static_cast<const std::int64_t*>(b), static_cast<std::int64_t*>(output), count); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void DivCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    switch (dtype) {
        case Dtype::kFloat16: HalfBinaryCPU<Float16, std::divides<float>>(a, b, output, count); break;
        case Dtype::kBFloat16: HalfBinaryCPU<BFloat16, std::divides<float>>(a, b, output, count); break;
        case Dtype::kFloat32: DivTypedCPU(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(output), count); break;
        case Dtype::kInt8: DivTypedCPU(static_cast<const std::int8_t*>(a), static_cast<const std::int8_t*>(b), static_cast<std::int8_t*>(output), count); break;
        case Dtype::kUInt8: DivTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<const std::uint8_t*>(b), static_cast<std::uint8_t*>(output), count); break;
        case Dtype::kInt16: DivTypedCPU(static_cast<const std::int16_t*>(a), static_cast<const std::int16_t*>(b), static_cast<std::int16_t*>(output), count); break;
        case Dtype::kInt32: DivTypedCPU(static_cast<const std::int32_t*>(a), static_cast<const std::int32_t*>(b), static_cast<std::int32_t*>(output), count); break;
        case Dtype::kInt64: DivTypedCPU(static_cast<const std::int64_t*>(a), static_cast<const std::int64_t*>(b), static_cast<std::int64_t*>(output), count); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void NegCPU(const void* a, void* output, std::size_t count, Dtype dtype) {
    switch (dtype) {
        case Dtype::kFloat16:
        case Dtype::kBFloat16: NegHalfCPU(a, output, count); break;
        case Dtype::kFloat32: NegTypedCPU(static_cast<const float*>(a), static_cast<float*>(output), count); break;
        case Dtype::kInt8: NegTypedCPU(static_cast<const std::int8_t*>(a), static_cast<std::int8_t*>(output), count); break;
        case Dtype::kUInt8: NegTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<std::uint8_t*>(output), count); break;
        case Dtype::kInt16: NegTypedCPU(static_cast<const std::int16_t*>(a), static_cast<std::int16_t*>(output), count); break;
        case Dtype::kInt32: NegTypedCPU(static_cast<const std::int32_t*>(a), static_cast<std::int32_t*>(output), count); break;
        case Dtype::kInt64: NegTypedCPU(static_cast<const std::int64_t*>(a), static_cast<std::int64_t*>(output), count); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

}  // namespace mtensor
