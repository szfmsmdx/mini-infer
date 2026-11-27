#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ops/matmul/kernels.h"

namespace mtensor {
namespace {

void MatmulFloat(const void* a, const void* b, void* output, int m, int k, int n,
                 StrideView a_stride,
                 StrideView b_stride) {
    const auto* a_data = static_cast<const float*>(a);
    const auto* b_data = static_cast<const float*>(b);
    auto* output_data = static_cast<float*>(output);
    for (int i = 0; i < m; ++i) {
        auto* output_row = output_data + static_cast<std::size_t>(i) * n;
        std::fill_n(output_row, n, 0.0f);
        for (int p = 0; p < k; ++p) {
            const float a_value = a_data[i * a_stride[0] + p * a_stride[1]];
            const auto* b_row = b_data + p * b_stride[0];
            for (int j = 0; j < n; ++j) {
                output_row[j] += a_value * b_row[j * b_stride[1]];
            }
        }
    }
}

template <typename U>
void MatmulInteger(const void* a, const void* b, void* output, int m, int k, int n,
                   StrideView a_stride,
                   StrideView b_stride) {
    // 用对应的无符号类型访问有符号整数的存储，实现按位宽回绕的运算。
    using Acc = std::conditional_t<(sizeof(U) < sizeof(std::uint32_t)), std::uint32_t, U>;
    const auto* a_data = static_cast<const U*>(a);
    const auto* b_data = static_cast<const U*>(b);
    auto* output_data = static_cast<U*>(output);
    for (int i = 0; i < m; ++i) {
        auto* output_row = output_data + static_cast<std::size_t>(i) * n;
        std::fill_n(output_row, n, U{0});
        for (int p = 0; p < k; ++p) {
            const Acc a_value = a_data[i * a_stride[0] + p * a_stride[1]];
            const auto* b_row = b_data + p * b_stride[0];
            for (int j = 0; j < n; ++j) {
                output_row[j] = static_cast<U>(output_row[j] + a_value * b_row[j * b_stride[1]]);
            }
        }
    }
}

template <float (*Decode)(const void*), std::uint16_t (*Encode)(float)>
void MatmulHalf(const void* a, const void* b, void* output, int m, int k, int n,
                StrideView a_stride,
                StrideView b_stride) {
    const auto* a_data = static_cast<const std::uint16_t*>(a);
    const auto* b_data = static_cast<const std::uint16_t*>(b);
    auto* output_data = static_cast<std::uint16_t*>(output);
    std::vector<float> row(n);
    for (int i = 0; i < m; ++i) {
        std::fill(row.begin(), row.end(), 0.0f);
        for (int p = 0; p < k; ++p) {
            const float a_value = Decode(a_data + i * a_stride[0] + p * a_stride[1]);
            const auto* b_row = b_data + p * b_stride[0];
            for (int j = 0; j < n; ++j) {
                row[j] += a_value * Decode(b_row + j * b_stride[1]);
            }
        }
        auto* output_row = output_data + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) {
            output_row[j] = Encode(row[j]);
        }
    }
}

}  // namespace

void MatmulCPU(const void* a, const void* b, void* output, Dtype dtype,
               int m, int k, int n, StrideView a_stride,
               StrideView b_stride) {
    if (m == 0 || n == 0) return;

    switch (dtype) {
        case Dtype::kFloat16:
            MatmulHalf<DecodeFloat16, EncodeFloat16>(a, b, output, m, k, n, a_stride, b_stride);
            return;
        case Dtype::kBFloat16:
            MatmulHalf<DecodeBFloat16, EncodeBFloat16>(a, b, output, m, k, n, a_stride, b_stride);
            return;
        case Dtype::kInt8:
        case Dtype::kUInt8:
            MatmulInteger<std::uint8_t>(a, b, output, m, k, n, a_stride, b_stride);
            return;
        case Dtype::kInt16:
            MatmulInteger<std::uint16_t>(a, b, output, m, k, n, a_stride, b_stride);
            return;
        case Dtype::kInt32:
            MatmulInteger<std::uint32_t>(a, b, output, m, k, n, a_stride, b_stride);
            return;
        case Dtype::kInt64:
            MatmulInteger<std::uint64_t>(a, b, output, m, k, n, a_stride, b_stride);
            return;
        case Dtype::kFloat32:
            MatmulFloat(a, b, output, m, k, n, a_stride, b_stride);
            return;
        default:
            throw std::invalid_argument("MatmulCPU: unsupported dtype");
    }
}

}  // namespace mtensor
