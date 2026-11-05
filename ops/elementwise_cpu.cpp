#include <cstddef>
#include <cstdint>
#include <cstring>
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

// FP16/BF16 共用 16 位编码：逐元素解码为 float 计算，再编码写回。
void AddHalfCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    auto* output_ptr = static_cast<std::uint16_t*>(output);
    const auto* a_ptr = static_cast<const std::uint16_t*>(a);
    const auto* b_ptr = static_cast<const std::uint16_t*>(b);

    if (dtype == Dtype::kBFloat16) {
        for (std::size_t i = 0; i < count; ++i) {
            const float result =
                DecodeBFloat16(a_ptr + i) + DecodeBFloat16(b_ptr + i);
            output_ptr[i] = EncodeBFloat16(result);
        }
    } else if (dtype == Dtype::kFloat16) {
        for (std::size_t i = 0; i < count; ++i) {
            const float result =
                DecodeFloat16(a_ptr + i) + DecodeFloat16(b_ptr + i);
            output_ptr[i] = EncodeFloat16(result);
        }
    } else {
        throw std::invalid_argument("AddHalfCPU: unsupported dtype");
    }
}

void SubHalfCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    auto* output_ptr = static_cast<std::uint16_t*>(output);
    const auto* a_ptr = static_cast<const std::uint16_t*>(a);
    const auto* b_ptr = static_cast<const std::uint16_t*>(b);

    if (dtype == Dtype::kBFloat16) {
        for (std::size_t i = 0; i < count; ++i) {
            const float result =
                DecodeBFloat16(a_ptr + i) - DecodeBFloat16(b_ptr + i);
            output_ptr[i] = EncodeBFloat16(result);
        }
    } else if (dtype == Dtype::kFloat16) {
        for (std::size_t i = 0; i < count; ++i) {
            const float result =
                DecodeFloat16(a_ptr + i) - DecodeFloat16(b_ptr + i);
            output_ptr[i] = EncodeFloat16(result);
        }
    } else {
        throw std::invalid_argument("SubHalfCPU: unsupported dtype");
    }
}

void MulHalfCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    auto* output_ptr = static_cast<std::uint16_t*>(output);
    const auto* a_ptr = static_cast<const std::uint16_t*>(a);
    const auto* b_ptr = static_cast<const std::uint16_t*>(b);

    if (dtype == Dtype::kBFloat16) {
        for (std::size_t i = 0; i < count; ++i) {
            const float result =
                DecodeBFloat16(a_ptr + i) * DecodeBFloat16(b_ptr + i);
            output_ptr[i] = EncodeBFloat16(result);
        }
    } else if (dtype == Dtype::kFloat16) {
        for (std::size_t i = 0; i < count; ++i) {
            const float result =
                DecodeFloat16(a_ptr + i) * DecodeFloat16(b_ptr + i);
            output_ptr[i] = EncodeFloat16(result);
        }
    } else {
        throw std::invalid_argument("MulHalfCPU: unsupported dtype");
    }
}

void DivHalfCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    auto* output_ptr = static_cast<std::uint16_t*>(output);
    const auto* a_ptr = static_cast<const std::uint16_t*>(a);
    const auto* b_ptr = static_cast<const std::uint16_t*>(b);

    if (dtype == Dtype::kBFloat16) {
        for (std::size_t i = 0; i < count; ++i) {
            const float result =
                DecodeBFloat16(a_ptr + i) / DecodeBFloat16(b_ptr + i);
            output_ptr[i] = EncodeBFloat16(result);
        }
    } else if (dtype == Dtype::kFloat16) {
        for (std::size_t i = 0; i < count; ++i) {
            const float result =
                DecodeFloat16(a_ptr + i) / DecodeFloat16(b_ptr + i);
            output_ptr[i] = EncodeFloat16(result);
        }
    } else {
        throw std::invalid_argument("DivHalfCPU: unsupported dtype");
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
    if (dtype == Dtype::kBFloat16 || dtype == Dtype::kFloat16) {
        AddHalfCPU(a, b, output, count, dtype);
        return;
    }

    DISPATCH_DTYPE(dtype, Element, {
        AddTypedCPU(
            static_cast<const Element*>(a),
            static_cast<const Element*>(b),
            static_cast<Element*>(output),
            count);
    });
}

void SubCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    if (dtype == Dtype::kBFloat16 || dtype == Dtype::kFloat16) {
        SubHalfCPU(a, b, output, count, dtype);
        return;
    }

    DISPATCH_DTYPE(dtype, Element, {
        SubTypedCPU(
            static_cast<const Element*>(a),
            static_cast<const Element*>(b),
            static_cast<Element*>(output),
            count);
    });
}

void MulCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    if (dtype == Dtype::kBFloat16 || dtype == Dtype::kFloat16) {
        MulHalfCPU(a, b, output, count, dtype);
        return;
    }

    DISPATCH_DTYPE(dtype, Element, {
        MulTypedCPU(
            static_cast<const Element*>(a),
            static_cast<const Element*>(b),
            static_cast<Element*>(output),
            count);
    });
}

void DivCPU(const void* a, const void* b, void* output, std::size_t count, Dtype dtype) {
    if (dtype == Dtype::kBFloat16 || dtype == Dtype::kFloat16) {
        DivHalfCPU(a, b, output, count, dtype);
        return;
    }

    DISPATCH_DTYPE(dtype, Element, {
        DivTypedCPU(
            static_cast<const Element*>(a),
            static_cast<const Element*>(b),
            static_cast<Element*>(output),
            count);
    });
}

void NegCPU(const void* a, void* output, std::size_t count, Dtype dtype) {
    if (dtype == Dtype::kBFloat16 || dtype == Dtype::kFloat16) {
        NegHalfCPU(a, output, count);
        return;
    }

    DISPATCH_DTYPE(dtype, Element, {
        NegTypedCPU(
            static_cast<const Element*>(a),
            static_cast<Element*>(output),
            count);
    });
}

}  // namespace mtensor
