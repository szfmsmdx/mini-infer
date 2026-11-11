#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ops/backend.h"

namespace mtensor {
namespace {

// 将连续的逻辑下标转换成实际存储中的元素偏移。
std::size_t ElementOffset(std::size_t index, const std::vector<int>& shape,
                          const std::vector<std::size_t>& stride) {
    std::size_t offset = 0;
    for (std::size_t i = shape.size(); i > 0; --i) {
        const std::size_t dimension = static_cast<std::size_t>(shape[i - 1]);
        offset += (index % dimension) * stride[i - 1];
        index /= dimension;
    }
    return offset;
}

}  // namespace

void SumCPU(const void* input, void* output, Dtype dtype, const std::vector<int>& shape,
            const std::vector<std::size_t>& stride, std::size_t numel) {
    const auto* bytes = static_cast<const std::byte*>(input);
    float float_sum = 0.0f;
    // 用无符号加法实现 Int64 固定位宽回绕，避免有符号溢出。
    std::uint64_t integer_sum = 0;

    for (std::size_t i = 0; i < numel; ++i) {
        const std::size_t offset = ElementOffset(i, shape, stride);
        const void* element = bytes + offset * dtype_size(dtype);
        switch (dtype) {
            case Dtype::kFloat32: float_sum += *static_cast<const float*>(element); break;
            case Dtype::kFloat16: float_sum += DecodeFloat16(element); break;
            case Dtype::kBFloat16: float_sum += DecodeBFloat16(element); break;
            case Dtype::kInt8:
                integer_sum += static_cast<std::uint64_t>(*static_cast<const std::int8_t*>(element)); break;
            case Dtype::kUInt8: integer_sum += *static_cast<const std::uint8_t*>(element); break;
            case Dtype::kInt16:
                integer_sum += static_cast<std::uint64_t>(*static_cast<const std::int16_t*>(element)); break;
            case Dtype::kInt32:
                integer_sum += static_cast<std::uint64_t>(*static_cast<const std::int32_t*>(element)); break;
            case Dtype::kInt64:
                integer_sum += static_cast<std::uint64_t>(*static_cast<const std::int64_t*>(element)); break;
            default: throw std::invalid_argument("sum: unsupported dtype");
        }
    }

    // 浮点输出保留输入 dtype；整数输出固定为 Int64。
    switch (dtype) {
        case Dtype::kFloat32: *static_cast<float*>(output) = float_sum; break;
        case Dtype::kFloat16: {
            const std::uint16_t bits = EncodeFloat16(float_sum);
            std::memcpy(output, &bits, sizeof(bits));
            break;
        }
        case Dtype::kBFloat16: {
            const std::uint16_t bits = EncodeBFloat16(float_sum);
            std::memcpy(output, &bits, sizeof(bits));
            break;
        }
        default: std::memcpy(output, &integer_sum, sizeof(integer_sum)); break;
    }
}

}  // namespace mtensor
