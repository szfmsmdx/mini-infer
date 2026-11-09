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

template <typename T>
void AddTypedCPU(
    const T* a, const T* b, T* output,
    const std::vector<int>& shape,
    const std::vector<std::size_t>& a_stride,
    const std::vector<std::size_t>& b_stride,
    const std::vector<std::size_t>& output_stride
) {
    auto dfs = [&](
        auto&& self,
        std::size_t dim,
        std::size_t a_offset, std::size_t b_offset, std::size_t output_offset
    ) -> void {
        if(dim == shape.size()) {
            if constexpr (std::is_integral<T>::value){
                *(output + output_offset) = AddWrap(*(a + a_offset), *(b + b_offset));
            }else{
                *(output + output_offset) = *(a + a_offset) + *(b + b_offset);
            }
            return;
        }

        for (int d = 0; d < shape[dim]; ++d){
            const std::size_t next_a_offset = a_offset + static_cast<std::size_t>(d) * a_stride[dim];
            const std::size_t next_b_offset = b_offset + static_cast<std::size_t>(d) * b_stride[dim];
            const std::size_t next_output_offset = output_offset + static_cast<std::size_t>(d) * output_stride[dim];
            self(self, dim + 1, next_a_offset, next_b_offset, next_output_offset);
        }
    };
    dfs(dfs, 0, 0, 0, 0);
}

template <typename Half>
void AddHalfCPU(const void* a, const void* b, void* output,
                const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
                const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride) {
    const auto* a_bytes = static_cast<const std::byte*>(a);
    const auto* b_bytes = static_cast<const std::byte*>(b);
    auto* output_bytes = static_cast<std::byte*>(output);
    auto dfs = [&](auto&& self, std::size_t dim, std::size_t a_offset,
                   std::size_t b_offset, std::size_t output_offset) -> void {
        if (dim == shape.size()) {
            std::uint16_t a_bits, b_bits;
            std::memcpy(&a_bits, a_bytes + a_offset * sizeof(a_bits), sizeof(a_bits));
            std::memcpy(&b_bits, b_bytes + b_offset * sizeof(b_bits), sizeof(b_bits));
            const Half result(static_cast<float>(Half::from_bits(a_bits)) +
                              static_cast<float>(Half::from_bits(b_bits)));
            std::memcpy(output_bytes + output_offset * sizeof(result.bits), &result.bits, sizeof(result.bits));
            return;
        }
        for (int d = 0; d < shape[dim]; ++d) {
            const auto index = static_cast<std::size_t>(d);
            self(self, dim + 1, a_offset + index * a_stride[dim],
                 b_offset + index * b_stride[dim], output_offset + index * output_stride[dim]);
        }
    };
    dfs(dfs, 0, 0, 0, 0);
}

template <typename T>
void SubTypedCPU(
    const T* a, const T* b, T* output,
    const std::vector<int>& shape,
    const std::vector<std::size_t>& a_stride,
    const std::vector<std::size_t>& b_stride,
    const std::vector<std::size_t>& output_stride
) {
    auto dfs = [&](
        auto&& self,
        std::size_t dim,
        std::size_t a_offset, std::size_t b_offset, std::size_t output_offset
    ) -> void {
        if(dim == shape.size()) {
            if constexpr (std::is_integral<T>::value){
                *(output + output_offset) = SubWrap(*(a + a_offset), *(b + b_offset));
            }else{
                *(output + output_offset) = *(a + a_offset) - *(b + b_offset);
            }
            return;
        }

        for (int d = 0; d < shape[dim]; ++d){
            const std::size_t next_a_offset = a_offset + static_cast<std::size_t>(d) * a_stride[dim];
            const std::size_t next_b_offset = b_offset + static_cast<std::size_t>(d) * b_stride[dim];
            const std::size_t next_output_offset = output_offset + static_cast<std::size_t>(d) * output_stride[dim];
            self(self, dim + 1, next_a_offset, next_b_offset, next_output_offset);
        }
    };
    dfs(dfs, 0, 0, 0, 0);
}

template <typename Half>
void SubHalfCPU(const void* a, const void* b, void* output,
                const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
                const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride) {
    const auto* a_bytes = static_cast<const std::byte*>(a);
    const auto* b_bytes = static_cast<const std::byte*>(b);
    auto* output_bytes = static_cast<std::byte*>(output);
    auto dfs = [&](auto&& self, std::size_t dim, std::size_t a_offset,
                   std::size_t b_offset, std::size_t output_offset) -> void {
        if (dim == shape.size()) {
            std::uint16_t a_bits, b_bits;
            std::memcpy(&a_bits, a_bytes + a_offset * sizeof(a_bits), sizeof(a_bits));
            std::memcpy(&b_bits, b_bytes + b_offset * sizeof(b_bits), sizeof(b_bits));
            const Half result(static_cast<float>(Half::from_bits(a_bits)) -
                              static_cast<float>(Half::from_bits(b_bits)));
            std::memcpy(output_bytes + output_offset * sizeof(result.bits), &result.bits, sizeof(result.bits));
            return;
        }
        for (int d = 0; d < shape[dim]; ++d) {
            const auto index = static_cast<std::size_t>(d);
            self(self, dim + 1, a_offset + index * a_stride[dim],
                 b_offset + index * b_stride[dim], output_offset + index * output_stride[dim]);
        }
    };
    dfs(dfs, 0, 0, 0, 0);
}

template <typename T>
void MulTypedCPU(
    const T* a, const T* b, T* output,
    const std::vector<int>& shape,
    const std::vector<std::size_t>& a_stride,
    const std::vector<std::size_t>& b_stride,
    const std::vector<std::size_t>& output_stride
) {
    auto dfs = [&](
        auto&& self,
        std::size_t dim,
        std::size_t a_offset, std::size_t b_offset, std::size_t output_offset
    ) -> void {
        if(dim == shape.size()) {
            if constexpr (std::is_integral<T>::value){
                *(output + output_offset) = MulWrap(*(a + a_offset), *(b + b_offset));
            }else{
                *(output + output_offset) = *(a + a_offset) * *(b + b_offset);
            }
            return;
        }

        for (int d = 0; d < shape[dim]; ++d){
            const std::size_t next_a_offset = a_offset + static_cast<std::size_t>(d) * a_stride[dim];
            const std::size_t next_b_offset = b_offset + static_cast<std::size_t>(d) * b_stride[dim];
            const std::size_t next_output_offset = output_offset + static_cast<std::size_t>(d) * output_stride[dim];
            self(self, dim + 1, next_a_offset, next_b_offset, next_output_offset);
        }
    };
    dfs(dfs, 0, 0, 0, 0);
}

template <typename Half>
void MulHalfCPU(const void* a, const void* b, void* output,
                const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
                const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride) {
    const auto* a_bytes = static_cast<const std::byte*>(a);
    const auto* b_bytes = static_cast<const std::byte*>(b);
    auto* output_bytes = static_cast<std::byte*>(output);
    auto dfs = [&](auto&& self, std::size_t dim, std::size_t a_offset,
                   std::size_t b_offset, std::size_t output_offset) -> void {
        if (dim == shape.size()) {
            std::uint16_t a_bits, b_bits;
            std::memcpy(&a_bits, a_bytes + a_offset * sizeof(a_bits), sizeof(a_bits));
            std::memcpy(&b_bits, b_bytes + b_offset * sizeof(b_bits), sizeof(b_bits));
            const Half result(static_cast<float>(Half::from_bits(a_bits)) *
                              static_cast<float>(Half::from_bits(b_bits)));
            std::memcpy(output_bytes + output_offset * sizeof(result.bits), &result.bits, sizeof(result.bits));
            return;
        }
        for (int d = 0; d < shape[dim]; ++d) {
            const auto index = static_cast<std::size_t>(d);
            self(self, dim + 1, a_offset + index * a_stride[dim],
                 b_offset + index * b_stride[dim], output_offset + index * output_stride[dim]);
        }
    };
    dfs(dfs, 0, 0, 0, 0);
}

template <typename T>
void DivTypedCPU(
    const T* a, const T* b, T* output,
    const std::vector<int>& shape,
    const std::vector<std::size_t>& a_stride,
    const std::vector<std::size_t>& b_stride,
    const std::vector<std::size_t>& output_stride
) {
    auto dfs = [&](
        auto&& self,
        std::size_t dim,
        std::size_t a_offset, std::size_t b_offset, std::size_t output_offset
    ) -> void {
        if(dim == shape.size()) {
            if constexpr (std::is_integral<T>::value){
                *(output + output_offset) = DivValue(*(a + a_offset), *(b + b_offset));
            }else{
                *(output + output_offset) = *(a + a_offset) / *(b + b_offset);
            }
            return;
        }

        for (int d = 0; d < shape[dim]; ++d){
            const std::size_t next_a_offset = a_offset + static_cast<std::size_t>(d) * a_stride[dim];
            const std::size_t next_b_offset = b_offset + static_cast<std::size_t>(d) * b_stride[dim];
            const std::size_t next_output_offset = output_offset + static_cast<std::size_t>(d) * output_stride[dim];
            self(self, dim + 1, next_a_offset, next_b_offset, next_output_offset);
        }
    };
    dfs(dfs, 0, 0, 0, 0);
}

template <typename Half>
void DivHalfCPU(const void* a, const void* b, void* output,
                const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
                const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride) {
    const auto* a_bytes = static_cast<const std::byte*>(a);
    const auto* b_bytes = static_cast<const std::byte*>(b);
    auto* output_bytes = static_cast<std::byte*>(output);
    auto dfs = [&](auto&& self, std::size_t dim, std::size_t a_offset,
                   std::size_t b_offset, std::size_t output_offset) -> void {
        if (dim == shape.size()) {
            std::uint16_t a_bits, b_bits;
            std::memcpy(&a_bits, a_bytes + a_offset * sizeof(a_bits), sizeof(a_bits));
            std::memcpy(&b_bits, b_bytes + b_offset * sizeof(b_bits), sizeof(b_bits));
            const Half result(static_cast<float>(Half::from_bits(a_bits)) /
                              static_cast<float>(Half::from_bits(b_bits)));
            std::memcpy(output_bytes + output_offset * sizeof(result.bits), &result.bits, sizeof(result.bits));
            return;
        }
        for (int d = 0; d < shape[dim]; ++d) {
            const auto index = static_cast<std::size_t>(d);
            self(self, dim + 1, a_offset + index * a_stride[dim],
                 b_offset + index * b_stride[dim], output_offset + index * output_stride[dim]);
        }
    };
    dfs(dfs, 0, 0, 0, 0);
}

template <typename T>
void NegTypedCPU(const T* a, T* output, const std::vector<int>& shape,
                 const std::vector<std::size_t>& a_stride,
                 const std::vector<std::size_t>& output_stride) {
    auto dfs = [&](auto&& self, std::size_t dim, std::size_t a_offset, std::size_t output_offset) -> void {
        if (dim == shape.size()) {
            output[output_offset] = NegValue(a[a_offset]);
            return;
        }
        for (int d = 0; d < shape[dim]; ++d) {
            const auto index = static_cast<std::size_t>(d);
            self(self, dim + 1, a_offset + index * a_stride[dim],
                 output_offset + index * output_stride[dim]);
        }
    };
    dfs(dfs, 0, 0, 0);
}

void NegHalfCPU(const void* a, void* output, const std::vector<int>& shape,
                const std::vector<std::size_t>& a_stride,
                const std::vector<std::size_t>& output_stride) {
    const auto* a_bytes = static_cast<const std::byte*>(a);
    auto* output_bytes = static_cast<std::byte*>(output);
    auto dfs = [&](auto&& self, std::size_t dim, std::size_t a_offset, std::size_t output_offset) -> void {
        if (dim == shape.size()) {
            std::uint16_t bits;
            std::memcpy(&bits, a_bytes + a_offset * sizeof(bits), sizeof(bits));
            bits ^= 0x8000u;
            std::memcpy(output_bytes + output_offset * sizeof(bits), &bits, sizeof(bits));
            return;
        }
        for (int d = 0; d < shape[dim]; ++d) {
            const auto index = static_cast<std::size_t>(d);
            self(self, dim + 1, a_offset + index * a_stride[dim],
                 output_offset + index * output_stride[dim]);
        }
    };
    dfs(dfs, 0, 0, 0);
}
}  // namespace

void AddCPU(const void* a, const void* b, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride) {
    switch (dtype) {
        case Dtype::kFloat16: AddHalfCPU<Float16>(a, b, output, shape, a_stride, b_stride, output_stride); break;
        case Dtype::kBFloat16: AddHalfCPU<BFloat16>(a, b, output, shape, a_stride, b_stride, output_stride); break;
        case Dtype::kFloat32: AddTypedCPU(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt8: AddTypedCPU(static_cast<const std::int8_t*>(a), static_cast<const std::int8_t*>(b), static_cast<std::int8_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kUInt8: AddTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<const std::uint8_t*>(b), static_cast<std::uint8_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt16: AddTypedCPU(static_cast<const std::int16_t*>(a), static_cast<const std::int16_t*>(b), static_cast<std::int16_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt32: AddTypedCPU(static_cast<const std::int32_t*>(a), static_cast<const std::int32_t*>(b), static_cast<std::int32_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt64: AddTypedCPU(static_cast<const std::int64_t*>(a), static_cast<const std::int64_t*>(b), static_cast<std::int64_t*>(output), shape, a_stride, b_stride, output_stride); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void SubCPU(const void* a, const void* b, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride) {
    switch (dtype) {
        case Dtype::kFloat16: SubHalfCPU<Float16>(a, b, output, shape, a_stride, b_stride, output_stride); break;
        case Dtype::kBFloat16: SubHalfCPU<BFloat16>(a, b, output, shape, a_stride, b_stride, output_stride); break;
        case Dtype::kFloat32: SubTypedCPU(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt8: SubTypedCPU(static_cast<const std::int8_t*>(a), static_cast<const std::int8_t*>(b), static_cast<std::int8_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kUInt8: SubTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<const std::uint8_t*>(b), static_cast<std::uint8_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt16: SubTypedCPU(static_cast<const std::int16_t*>(a), static_cast<const std::int16_t*>(b), static_cast<std::int16_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt32: SubTypedCPU(static_cast<const std::int32_t*>(a), static_cast<const std::int32_t*>(b), static_cast<std::int32_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt64: SubTypedCPU(static_cast<const std::int64_t*>(a), static_cast<const std::int64_t*>(b), static_cast<std::int64_t*>(output), shape, a_stride, b_stride, output_stride); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void MulCPU(const void* a, const void* b, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride) {
    switch (dtype) {
        case Dtype::kFloat16: MulHalfCPU<Float16>(a, b, output, shape, a_stride, b_stride, output_stride); break;
        case Dtype::kBFloat16: MulHalfCPU<BFloat16>(a, b, output, shape, a_stride, b_stride, output_stride); break;
        case Dtype::kFloat32: MulTypedCPU(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt8: MulTypedCPU(static_cast<const std::int8_t*>(a), static_cast<const std::int8_t*>(b), static_cast<std::int8_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kUInt8: MulTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<const std::uint8_t*>(b), static_cast<std::uint8_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt16: MulTypedCPU(static_cast<const std::int16_t*>(a), static_cast<const std::int16_t*>(b), static_cast<std::int16_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt32: MulTypedCPU(static_cast<const std::int32_t*>(a), static_cast<const std::int32_t*>(b), static_cast<std::int32_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt64: MulTypedCPU(static_cast<const std::int64_t*>(a), static_cast<const std::int64_t*>(b), static_cast<std::int64_t*>(output), shape, a_stride, b_stride, output_stride); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void DivCPU(const void* a, const void* b, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& b_stride, const std::vector<std::size_t>& output_stride) {
    switch (dtype) {
        case Dtype::kFloat16: DivHalfCPU<Float16>(a, b, output, shape, a_stride, b_stride, output_stride); break;
        case Dtype::kBFloat16: DivHalfCPU<BFloat16>(a, b, output, shape, a_stride, b_stride, output_stride); break;
        case Dtype::kFloat32: DivTypedCPU(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt8: DivTypedCPU(static_cast<const std::int8_t*>(a), static_cast<const std::int8_t*>(b), static_cast<std::int8_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kUInt8: DivTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<const std::uint8_t*>(b), static_cast<std::uint8_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt16: DivTypedCPU(static_cast<const std::int16_t*>(a), static_cast<const std::int16_t*>(b), static_cast<std::int16_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt32: DivTypedCPU(static_cast<const std::int32_t*>(a), static_cast<const std::int32_t*>(b), static_cast<std::int32_t*>(output), shape, a_stride, b_stride, output_stride); break;
        case Dtype::kInt64: DivTypedCPU(static_cast<const std::int64_t*>(a), static_cast<const std::int64_t*>(b), static_cast<std::int64_t*>(output), shape, a_stride, b_stride, output_stride); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void NegCPU(const void* a, void* output, Dtype dtype,
            const std::vector<int>& shape, const std::vector<std::size_t>& a_stride,
            const std::vector<std::size_t>& output_stride) {
    switch (dtype) {
        case Dtype::kFloat16:
        case Dtype::kBFloat16: NegHalfCPU(a, output, shape, a_stride, output_stride); break;
        case Dtype::kFloat32: NegTypedCPU(static_cast<const float*>(a), static_cast<float*>(output), shape, a_stride, output_stride); break;
        case Dtype::kInt8: NegTypedCPU(static_cast<const std::int8_t*>(a), static_cast<std::int8_t*>(output), shape, a_stride, output_stride); break;
        case Dtype::kUInt8: NegTypedCPU(static_cast<const std::uint8_t*>(a), static_cast<std::uint8_t*>(output), shape, a_stride, output_stride); break;
        case Dtype::kInt16: NegTypedCPU(static_cast<const std::int16_t*>(a), static_cast<std::int16_t*>(output), shape, a_stride, output_stride); break;
        case Dtype::kInt32: NegTypedCPU(static_cast<const std::int32_t*>(a), static_cast<std::int32_t*>(output), shape, a_stride, output_stride); break;
        case Dtype::kInt64: NegTypedCPU(static_cast<const std::int64_t*>(a), static_cast<std::int64_t*>(output), shape, a_stride, output_stride); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

}  // namespace mtensor
