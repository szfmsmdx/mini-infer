#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "ops/elementwise/kernels.h"
#include "backend/common/cpu_iteration.h"
#include "mini_infer/ops/elementwise.h"

namespace mtensor {
namespace {

// 整数按固定位宽回绕。先转无符号计算，避免有符号加法溢出。
template <typename T>
T AddWrap(T a, T b) {
    using U = std::make_unsigned_t<T>;
    const U lhs = static_cast<U>(a);
    const U rhs = static_cast<U>(b);
    const U sum = lhs + rhs;

    T result;
    std::memcpy(&result, &sum, sizeof(result));
    return result;
}

template <typename T>
T SubWrap(T a, T b) {
    using U = std::make_unsigned_t<T>;
    const U lhs = static_cast<U>(a);
    const U rhs = static_cast<U>(b);
    const U diff = lhs - rhs;

    T result;
    std::memcpy(&result, &diff, sizeof(result));
    return result;
}

template <typename T>
T MulWrap(T a, T b) {
    using U = std::make_unsigned_t<T>;
    // 无符号乘法保留低位，也避免 8/16 位运算提升为 int 后溢出。
    const auto lhs = static_cast<std::uint64_t>(a);
    const auto rhs = static_cast<std::uint64_t>(b);
    const U product = static_cast<U>(lhs * rhs);

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
    const void* a, const void* b, void* output,
    ShapeView shape,
    StrideView a_stride,
    StrideView b_stride,
    StrideView output_stride
) {
    const auto* lhs = static_cast<const T*>(a);
    const auto* rhs = static_cast<const T*>(b);
    auto* result = static_cast<T*>(output);
    ForEachOffset(shape, a_stride, b_stride, output_stride,
        [&](std::size_t a_offset, std::size_t b_offset, std::size_t output_offset) {
            if constexpr (std::is_integral<T>::value){
                result[output_offset] = AddWrap(lhs[a_offset], rhs[b_offset]);
            }else{
                result[output_offset] = lhs[a_offset] + rhs[b_offset];
            }
        });
}

template <float (*Decode)(const void*), std::uint16_t (*Encode)(float)>
void AddHalfCPU(const void* a, const void* b, void* output,
                ShapeView shape, StrideView a_stride,
                StrideView b_stride, StrideView output_stride) {
    const auto* lhs = static_cast<const std::uint16_t*>(a);
    const auto* rhs = static_cast<const std::uint16_t*>(b);
    auto* result = static_cast<std::uint16_t*>(output);
    ForEachOffset(shape, a_stride, b_stride, output_stride,
        [&](std::size_t a_offset, std::size_t b_offset, std::size_t output_offset) {
            const float a_value = Decode(lhs + a_offset);
            const float b_value = Decode(rhs + b_offset);
            result[output_offset] = Encode(a_value + b_value);
        });
}

template <typename T>
void SubTypedCPU(
    const void* a, const void* b, void* output,
    ShapeView shape,
    StrideView a_stride,
    StrideView b_stride,
    StrideView output_stride
) {
    const auto* lhs = static_cast<const T*>(a);
    const auto* rhs = static_cast<const T*>(b);
    auto* result = static_cast<T*>(output);
    ForEachOffset(shape, a_stride, b_stride, output_stride,
        [&](std::size_t a_offset, std::size_t b_offset, std::size_t output_offset) {
            if constexpr (std::is_integral<T>::value){
                result[output_offset] = SubWrap(lhs[a_offset], rhs[b_offset]);
            }else{
                result[output_offset] = lhs[a_offset] - rhs[b_offset];
            }
        });
}

template <float (*Decode)(const void*), std::uint16_t (*Encode)(float)>
void SubHalfCPU(const void* a, const void* b, void* output,
                ShapeView shape, StrideView a_stride,
                StrideView b_stride, StrideView output_stride) {
    const auto* lhs = static_cast<const std::uint16_t*>(a);
    const auto* rhs = static_cast<const std::uint16_t*>(b);
    auto* result = static_cast<std::uint16_t*>(output);
    ForEachOffset(shape, a_stride, b_stride, output_stride,
        [&](std::size_t a_offset, std::size_t b_offset, std::size_t output_offset) {
            const float a_value = Decode(lhs + a_offset);
            const float b_value = Decode(rhs + b_offset);
            result[output_offset] = Encode(a_value - b_value);
        });
}

template <typename T>
void MulTypedCPU(
    const void* a, const void* b, void* output,
    ShapeView shape,
    StrideView a_stride,
    StrideView b_stride,
    StrideView output_stride
) {
    const auto* lhs = static_cast<const T*>(a);
    const auto* rhs = static_cast<const T*>(b);
    auto* result = static_cast<T*>(output);
    ForEachOffset(shape, a_stride, b_stride, output_stride,
        [&](std::size_t a_offset, std::size_t b_offset, std::size_t output_offset) {
            if constexpr (std::is_integral<T>::value){
                result[output_offset] = MulWrap(lhs[a_offset], rhs[b_offset]);
            }else{
                result[output_offset] = lhs[a_offset] * rhs[b_offset];
            }
        });
}

template <float (*Decode)(const void*), std::uint16_t (*Encode)(float)>
void MulHalfCPU(const void* a, const void* b, void* output,
                ShapeView shape, StrideView a_stride,
                StrideView b_stride, StrideView output_stride) {
    const auto* lhs = static_cast<const std::uint16_t*>(a);
    const auto* rhs = static_cast<const std::uint16_t*>(b);
    auto* result = static_cast<std::uint16_t*>(output);
    ForEachOffset(shape, a_stride, b_stride, output_stride,
        [&](std::size_t a_offset, std::size_t b_offset, std::size_t output_offset) {
            const float a_value = Decode(lhs + a_offset);
            const float b_value = Decode(rhs + b_offset);
            result[output_offset] = Encode(a_value * b_value);
        });
}

template <typename T>
void DivTypedCPU(
    const void* a, const void* b, void* output,
    ShapeView shape,
    StrideView a_stride,
    StrideView b_stride,
    StrideView output_stride
) {
    const auto* lhs = static_cast<const T*>(a);
    const auto* rhs = static_cast<const T*>(b);
    auto* result = static_cast<T*>(output);
    ForEachOffset(shape, a_stride, b_stride, output_stride,
        [&](std::size_t a_offset, std::size_t b_offset, std::size_t output_offset) {
            if constexpr (std::is_integral<T>::value){
                result[output_offset] = DivValue(lhs[a_offset], rhs[b_offset]);
            }else{
                result[output_offset] = lhs[a_offset] / rhs[b_offset];
            }
        });
}

template <float (*Decode)(const void*), std::uint16_t (*Encode)(float)>
void DivHalfCPU(const void* a, const void* b, void* output,
                ShapeView shape, StrideView a_stride,
                StrideView b_stride, StrideView output_stride) {
    const auto* lhs = static_cast<const std::uint16_t*>(a);
    const auto* rhs = static_cast<const std::uint16_t*>(b);
    auto* result = static_cast<std::uint16_t*>(output);
    ForEachOffset(shape, a_stride, b_stride, output_stride,
        [&](std::size_t a_offset, std::size_t b_offset, std::size_t output_offset) {
            const float a_value = Decode(lhs + a_offset);
            const float b_value = Decode(rhs + b_offset);
            result[output_offset] = Encode(a_value / b_value);
        });
}

template <typename T>
void NegTypedCPU(const void* a, void* output, ShapeView shape,
                 StrideView a_stride,
                 StrideView output_stride) {
    const auto* input = static_cast<const T*>(a);
    auto* result = static_cast<T*>(output);
    ForEachOffset(shape, a_stride, output_stride,
        [&](std::size_t a_offset, std::size_t output_offset) {
            result[output_offset] = NegValue(input[a_offset]);
        });
}

void NegHalfCPU(const void* a, void* output, ShapeView shape,
                StrideView a_stride,
                StrideView output_stride) {
    const auto* a_bytes = static_cast<const std::byte*>(a);
    auto* output_bytes = static_cast<std::byte*>(output);
    ForEachOffset(shape, a_stride, output_stride,
        [&](std::size_t a_offset, std::size_t output_offset) {
            std::uint16_t bits;
            std::memcpy(&bits, a_bytes + a_offset * sizeof(bits), sizeof(bits));
            bits ^= 0x8000u;
            std::memcpy(output_bytes + output_offset * sizeof(bits), &bits, sizeof(bits));
        });
}
template <typename Input, typename Output>
void SqrtTypedCPU(const void* input, void* output, ShapeView shape,
                  StrideView stride) {
    const auto* src = static_cast<const std::byte*>(input);
    auto* dst = static_cast<std::byte*>(output);
    ForEachOffset(shape, stride,
        [&](std::size_t offset) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const Output root(std::sqrt(static_cast<float>(value)));
            std::memcpy(dst, &root, sizeof(Output));
            dst += sizeof(Output);
        });
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
    ForEachOffset(x.shape(), x.stride(),
        [&](std::size_t xi) {
            write(index++, xi);
        });
}

template <typename Input, typename Output, bool Log>
void ExpLogTypedCPU(const Tensor& x, Tensor& out) {
    const auto* src = static_cast<const std::byte*>(x.data_ptr());
    auto* dst = static_cast<std::byte*>(out.data_ptr());
    const auto& shape = x.shape();
    const auto& stride = x.stride();
    ForEachOffset(shape, stride,
        [&](std::size_t offset) {
            Input value(0);
            std::memcpy(&value, src + offset * sizeof(Input), sizeof(Input));
            const float v = static_cast<float>(value);
            const Output result(Log ? std::log(v) : std::exp(v));
            std::memcpy(dst, &result, sizeof(Output));
            dst += sizeof(Output);
        });
}

}  // namespace

void AddCPU(const void* a, const void* b, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView b_stride, StrideView output_stride) {
    switch (dtype) {
        case Dtype::kFloat16:
            AddHalfCPU<DecodeFloat16, EncodeFloat16>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kBFloat16:
            AddHalfCPU<DecodeBFloat16, EncodeBFloat16>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kFloat32:
            AddTypedCPU<float>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt8:
            AddTypedCPU<std::int8_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kUInt8:
            AddTypedCPU<std::uint8_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt16:
            AddTypedCPU<std::int16_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt32:
            AddTypedCPU<std::int32_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt64:
            AddTypedCPU<std::int64_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void SubCPU(const void* a, const void* b, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView b_stride, StrideView output_stride) {
    switch (dtype) {
        case Dtype::kFloat16:
            SubHalfCPU<DecodeFloat16, EncodeFloat16>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kBFloat16:
            SubHalfCPU<DecodeBFloat16, EncodeBFloat16>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kFloat32:
            SubTypedCPU<float>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt8:
            SubTypedCPU<std::int8_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kUInt8:
            SubTypedCPU<std::uint8_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt16:
            SubTypedCPU<std::int16_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt32:
            SubTypedCPU<std::int32_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt64:
            SubTypedCPU<std::int64_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void MulCPU(const void* a, const void* b, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView b_stride, StrideView output_stride) {
    switch (dtype) {
        case Dtype::kFloat16:
            MulHalfCPU<DecodeFloat16, EncodeFloat16>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kBFloat16:
            MulHalfCPU<DecodeBFloat16, EncodeBFloat16>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kFloat32:
            MulTypedCPU<float>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt8:
            MulTypedCPU<std::int8_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kUInt8:
            MulTypedCPU<std::uint8_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt16:
            MulTypedCPU<std::int16_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt32:
            MulTypedCPU<std::int32_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt64:
            MulTypedCPU<std::int64_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void DivCPU(const void* a, const void* b, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView b_stride, StrideView output_stride) {
    switch (dtype) {
        case Dtype::kFloat16:
            DivHalfCPU<DecodeFloat16, EncodeFloat16>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kBFloat16:
            DivHalfCPU<DecodeBFloat16, EncodeBFloat16>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kFloat32:
            DivTypedCPU<float>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt8:
            DivTypedCPU<std::int8_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kUInt8:
            DivTypedCPU<std::uint8_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt16:
            DivTypedCPU<std::int16_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt32:
            DivTypedCPU<std::int32_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        case Dtype::kInt64:
            DivTypedCPU<std::int64_t>(a, b, output, shape, a_stride, b_stride, output_stride);
            break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void NegCPU(const void* a, void* output, Dtype dtype,
            ShapeView shape, StrideView a_stride,
            StrideView output_stride) {
    switch (dtype) {
        case Dtype::kFloat16:
        case Dtype::kBFloat16: NegHalfCPU(a, output, shape, a_stride, output_stride); break;
        case Dtype::kFloat32:
            NegTypedCPU<float>(a, output, shape, a_stride, output_stride);
            break;
        case Dtype::kInt8:
            NegTypedCPU<std::int8_t>(a, output, shape, a_stride, output_stride);
            break;
        case Dtype::kUInt8:
            NegTypedCPU<std::uint8_t>(a, output, shape, a_stride, output_stride);
            break;
        case Dtype::kInt16:
            NegTypedCPU<std::int16_t>(a, output, shape, a_stride, output_stride);
            break;
        case Dtype::kInt32:
            NegTypedCPU<std::int32_t>(a, output, shape, a_stride, output_stride);
            break;
        case Dtype::kInt64:
            NegTypedCPU<std::int64_t>(a, output, shape, a_stride, output_stride);
            break;
        default: throw std::invalid_argument("unsupported dtype");
    }
}

void SqrtCPU(const void* input, void* output, Dtype dtype, ShapeView shape,
             StrideView stride) {
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

template <bool Log>
static void ExpLogCPU(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: ExpLogTypedCPU<float, float, Log>(x, out); break;
        case Dtype::kFloat16: ExpLogTypedCPU<Float16, Float16, Log>(x, out); break;
        case Dtype::kBFloat16: ExpLogTypedCPU<BFloat16, BFloat16, Log>(x, out); break;
        case Dtype::kInt8: ExpLogTypedCPU<std::int8_t, float, Log>(x, out); break;
        case Dtype::kUInt8: ExpLogTypedCPU<std::uint8_t, float, Log>(x, out); break;
        case Dtype::kInt16: ExpLogTypedCPU<std::int16_t, float, Log>(x, out); break;
        case Dtype::kInt32: ExpLogTypedCPU<std::int32_t, float, Log>(x, out); break;
        case Dtype::kInt64: ExpLogTypedCPU<std::int64_t, float, Log>(x, out); break;
        default: throw std::invalid_argument(Log ? "log: unsupported dtype" : "exp: unsupported dtype");
    }
}

void ExpCPU(const Tensor& x, Tensor& out) {
    ExpLogCPU<false>(x, out);
}

void LogCPU(const Tensor& x, Tensor& out) {
    ExpLogCPU<true>(x, out);
}

namespace {

template <typename T>
void RsqrtBackwardTypedCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    const auto* input = static_cast<const std::byte*>(x.data_ptr());
    const auto* grad = static_cast<const std::byte*>(output_grad.data_ptr());
    auto* output = static_cast<std::byte*>(dx.data_ptr());
    ForEachOffset(x.shape(), x.stride(), output_grad.stride(),
        [&](std::size_t xi, std::size_t gi) {
            T value(0), upstream(0);
            std::memcpy(&value, input + xi * sizeof(T), sizeof(T));
            std::memcpy(&upstream, grad + gi * sizeof(T), sizeof(T));
            const float y = 1.0f / std::sqrt(static_cast<float>(value));
            const T result(((-0.5f * y) * static_cast<float>(upstream)) * y * y);
            std::memcpy(output, &result, sizeof(T));
            output += sizeof(T);
        });
}

}  // namespace

void RsqrtBackwardCPU(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: RsqrtBackwardTypedCPU<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: RsqrtBackwardTypedCPU<Float16>(x, output_grad, dx); break;
        case Dtype::kBFloat16: RsqrtBackwardTypedCPU<BFloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("rsqrt backward: unsupported dtype");
    }
}

void SubBackwardCPU(const Tensor& grad, Tensor& db) {
    // 左侧梯度直接传回，这里只计算右侧。
    neg_out(grad, db);
}

void MulBackwardCPU(const Tensor& a, const Tensor& b, const Tensor& grad,
                    Tensor& da, Tensor& db) {
    mul_out(grad, b, da);
    mul_out(grad, a, db);
}

void DivBackwardCPU(const Tensor& a, const Tensor& b, const Tensor& grad,
                    Tensor& da, Tensor& db) {
    div_out(grad, b, da);
    mul_out(neg(grad), (a / b) / b, db);
}

void NegBackwardCPU(const Tensor& grad, Tensor& dx) {
    neg_out(grad, dx);
}

void SqrtBackwardCPU(const Tensor& x, const Tensor& grad, Tensor& dx) {
    const Tensor root = sqrt(x);
    div_out(grad, root + root, dx);
}

void ExpBackwardCPU(const Tensor& y, const Tensor& grad, Tensor& dx) {
    mul_out(grad, y, dx);
}

void LogBackwardCPU(const Tensor& x, const Tensor& grad, Tensor& dx) {
    div_out(grad, x, dx);
}

}  // namespace mtensor
