#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "ops/backend.h"
#include "mini_infer/tensor.h"

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
template <typename Input, typename Output>
void SqrtTypedCPU(const void* input, void* output, const std::vector<int>& shape,
                  const std::vector<std::size_t>& stride) {
    const auto* src = static_cast<const std::byte*>(input);
    auto* dst = static_cast<std::byte*>(output);
    auto visit = [&](auto&& self, std::size_t dim, std::size_t offset) -> void {
        if (dim == shape.size()) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const Output root(std::sqrt(static_cast<float>(value)));
            std::memcpy(dst, &root, sizeof(Output));
            dst += sizeof(Output);
            return;
        }
        for (int i = 0; i < shape[dim]; ++i) {
            self(self, dim + 1, offset + i * stride[dim]);
        }
    };
    visit(visit, 0, 0);
}

template <typename In, typename Out>
void RsqrtTypedCPU(const Tensor& x, Tensor& result) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(result.data_ptr());
    auto write = [&](std::size_t dst_index, std::size_t x_index) {
        In value(0);
        std::memcpy(&value, src + x_index * sizeof(In), sizeof(In));
        const float y = 1.0f / std::sqrt(static_cast<float>(value));
        const Out out(y);
        std::memcpy(dst + dst_index * sizeof(Out), &out, sizeof(Out));
    };
    if (x.is_contiguous()) {
        const auto count = x.numel();
        for (std::size_t i = 0; i < count; ++i) write(i, i);
        return;
    }
    std::size_t index = 0;
    auto visit = [&](auto&& self, int dim, std::size_t xi) -> void {
        if (dim == x.dim()) {
            write(index++, xi);
            return;
        }
        const auto xs = x.stride()[dim];
        for (int i = 0; i < x.size(dim); ++i) self(self, dim + 1, xi + i * xs);
    };
    visit(visit, 0, 0);
}

template <typename Input, typename Output>
void ExpTypedCPU(const Tensor& x, Tensor& out) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(out.data_ptr());
    const auto& shape = x.shape();
    const auto& stride = x.stride();
    auto visit = [&](auto&& self, std::size_t dim, std::size_t offset) -> void {
        if (dim == shape.size()) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const Output result(std::exp(static_cast<float>(value)));
            std::memcpy(dst, &result, sizeof(Output));
            dst += sizeof(Output);
            return;
        }
        for (int i = 0; i < shape[dim]; ++i) {
            self(self, dim + 1, offset + i * stride[dim]);
        }
    };
    visit(visit, 0, 0);
}

template <typename Input, typename Output>
void SigmoidTypedCPU(const Tensor& x, Tensor& out) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(out.data_ptr());
    const auto& shape = x.shape();
    const auto& stride = x.stride();
    auto visit = [&](auto&& self, std::size_t dim, std::size_t offset) -> void {
        if (dim == shape.size()) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const float v = static_cast<float>(value);
            const float t = std::exp(-std::abs(v));
            const Output result(v >= 0.0f ? 1.0f / (1.0f + t) : t / (1.0f + t));
            std::memcpy(dst, &result, sizeof(Output));
            dst += sizeof(Output);
            return;
        }
        for (int i = 0; i < shape[dim]; ++i) {
            self(self, dim + 1, offset + i * stride[dim]);
        }
    };
    visit(visit, 0, 0);
}

template <typename Input, typename Output>
void SiluTypedCPU(const Tensor& x, Tensor& out) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(out.data_ptr());
    const auto& shape = x.shape();
    const auto& stride = x.stride();
    auto visit = [&](auto&& self, std::size_t dim, std::size_t offset) -> void {
        if (dim == shape.size()) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const float v = static_cast<float>(value);
            float y;
            if (v < -80.0f) {
                // 负尾部用 double，避免 exp(x) 在乘 x 前下溢。
                const double t = std::exp(static_cast<double>(v));
                y = static_cast<float>(v * t / (1.0 + t));
            } else {
                const float t = std::exp(-std::abs(v));
                y = v >= 0.0f ? v / (1.0f + t) : (v * t) / (1.0f + t);
            }
            const Output result(y);
            std::memcpy(dst, &result, sizeof(Output));
            dst += sizeof(Output);
            return;
        }
        for (int i = 0; i < shape[dim]; ++i)
            self(self, dim + 1, offset + i * stride[dim]);
    };
    visit(visit, 0, 0);
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

void SqrtCPU(const void* input, void* output, Dtype dtype, const std::vector<int>& shape,
             const std::vector<std::size_t>& stride) {
    switch (dtype) {
        case Dtype::kFloat32: SqrtTypedCPU<float, float>(input, output, shape, stride); break;
        case Dtype::kFloat16: SqrtTypedCPU<Float16, Float16>(input, output, shape, stride); break;
        case Dtype::kBFloat16: SqrtTypedCPU<BFloat16, BFloat16>(input, output, shape, stride); break;
        case Dtype::kInt8: SqrtTypedCPU<std::int8_t, float>(input, output, shape, stride); break;
        case Dtype::kUInt8: SqrtTypedCPU<std::uint8_t, float>(input, output, shape, stride); break;
        case Dtype::kInt16: SqrtTypedCPU<std::int16_t, float>(input, output, shape, stride); break;
        case Dtype::kInt32: SqrtTypedCPU<std::int32_t, float>(input, output, shape, stride); break;
        case Dtype::kInt64: SqrtTypedCPU<std::int64_t, float>(input, output, shape, stride); break;
        default: throw std::invalid_argument("sqrt: unsupported dtype");
    }
}

void RsqrtCPU(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: RsqrtTypedCPU<float, float>(x, out); break;
        case Dtype::kFloat16: RsqrtTypedCPU<Float16, Float16>(x, out); break;
        case Dtype::kBFloat16: RsqrtTypedCPU<BFloat16, BFloat16>(x, out); break;
        case Dtype::kInt8: RsqrtTypedCPU<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: RsqrtTypedCPU<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: RsqrtTypedCPU<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: RsqrtTypedCPU<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: RsqrtTypedCPU<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("rsqrt: unsupported dtype");
    }
}

void ExpCPU(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: ExpTypedCPU<float, float>(x, out); break;
        case Dtype::kFloat16: ExpTypedCPU<Float16, Float16>(x, out); break;
        case Dtype::kBFloat16: ExpTypedCPU<BFloat16, BFloat16>(x, out); break;
        case Dtype::kInt8: ExpTypedCPU<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: ExpTypedCPU<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: ExpTypedCPU<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: ExpTypedCPU<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: ExpTypedCPU<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("exp: unsupported dtype");
    }
}

void SigmoidCPU(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: SigmoidTypedCPU<float, float>(x, out); break;
        case Dtype::kFloat16: SigmoidTypedCPU<Float16, Float16>(x, out); break;
        case Dtype::kBFloat16: SigmoidTypedCPU<BFloat16, BFloat16>(x, out); break;
        case Dtype::kInt8: SigmoidTypedCPU<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: SigmoidTypedCPU<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: SigmoidTypedCPU<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: SigmoidTypedCPU<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: SigmoidTypedCPU<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("sigmoid: unsupported dtype");
    }
}

void SiluCPU(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: SiluTypedCPU<float, float>(x, out); break;
        case Dtype::kFloat16: SiluTypedCPU<Float16, Float16>(x, out); break;
        case Dtype::kBFloat16: SiluTypedCPU<BFloat16, BFloat16>(x, out); break;
        case Dtype::kInt8: SiluTypedCPU<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: SiluTypedCPU<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: SiluTypedCPU<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: SiluTypedCPU<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: SiluTypedCPU<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("silu: unsupported dtype");
    }
}

}  // namespace mtensor
