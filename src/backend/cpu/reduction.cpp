#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mini_infer/tensor/tensor.h"
#include "mini_infer/ops/elementwise.h"
#include "ops/reduction/kernels.h"

namespace mtensor {
namespace {

// 将连续的逻辑下标转换成实际存储中的元素偏移。
std::size_t ElementOffset(std::size_t index, ShapeView shape,
                          StrideView stride) {
    std::size_t offset = 0;
    for (std::size_t i = shape.size(); i > 0; --i) {
        const std::size_t dimension = static_cast<std::size_t>(shape[i - 1]);
        offset += (index % dimension) * stride[i - 1];
        index /= dimension;
    }
    return offset;
}

float ReadFloat(const void* input, Dtype dtype, std::size_t offset) {
    switch (dtype) {
        case Dtype::kFloat32: return static_cast<const float*>(input)[offset];
        case Dtype::kFloat16: return DecodeFloat16(static_cast<const std::uint16_t*>(input) + offset);
        case Dtype::kBFloat16: return DecodeBFloat16(static_cast<const std::uint16_t*>(input) + offset);
        default: throw std::invalid_argument("extrema: unsupported floating dtype");
    }
}

std::int64_t ReadSigned(const void* input, Dtype dtype, std::size_t offset) {
    switch (dtype) {
        case Dtype::kInt8: return static_cast<const std::int8_t*>(input)[offset];
        case Dtype::kUInt8: return static_cast<const std::uint8_t*>(input)[offset];
        case Dtype::kInt16: return static_cast<const std::int16_t*>(input)[offset];
        case Dtype::kInt32: return static_cast<const std::int32_t*>(input)[offset];
        case Dtype::kInt64: return static_cast<const std::int64_t*>(input)[offset];
        default: throw std::invalid_argument("extrema: unsupported integer dtype");
    }
}

template <typename T>
T ReadExtrema(const void* input, Dtype dtype, std::size_t offset) {
    if constexpr (std::is_same_v<T, float>) return ReadFloat(input, dtype, offset);
    else return ReadSigned(input, dtype, offset);
}

// 将归约组的下标映射到输入的逻辑下标；组内以 inner_size 为步长。
std::size_t GroupStart(std::size_t group, std::size_t count, std::size_t inner_size) {
    return group / inner_size * count * inner_size + group % inner_size;
}

template <typename T>
void ExtremaTypedCPU(const Tensor& input, Tensor& output, std::size_t inner_size, bool maximum) {
    const auto* bytes = static_cast<const std::byte*>(input.data_ptr());
    auto* result = static_cast<std::byte*>(output.data_ptr());
    const std::size_t element_size = input.element_size();
    const std::size_t groups = output.numel();
    const std::size_t count = input.numel() / groups;
    const bool contiguous = input.is_contiguous();
    const auto offset_for = [&](std::size_t i) {
        return contiguous ? i : ElementOffset(i, input.shape(), input.stride());
    };
    for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t start = GroupStart(group, count, inner_size);
        std::size_t best_offset = offset_for(start);
        T best = ReadExtrema<T>(bytes, input.dtype(), best_offset);
        for (std::size_t j = 1; j < count; ++j) {
            if constexpr (std::is_same_v<T, float>) {
                if (std::isnan(best)) break;
            }
            const std::size_t offset = offset_for(start + j * inner_size);
            const T candidate = ReadExtrema<T>(bytes, input.dtype(), offset);
            bool better = maximum ? candidate > best : candidate < best;
            if constexpr (std::is_same_v<T, float>) better = better || std::isnan(candidate);
            if (better) {
                best = candidate;
                best_offset = offset;
            }
        }
        std::memcpy(result + group * element_size, bytes + best_offset * element_size, element_size);
    }
}

void WriteFloat(void* output, Dtype dtype, std::size_t offset, float value) {
    if (dtype == Dtype::kFloat32) {
        static_cast<float*>(output)[offset] = value;
    } else {
        const auto bits = dtype == Dtype::kFloat16 ? EncodeFloat16(value) : EncodeBFloat16(value);
        std::memcpy(static_cast<std::byte*>(output) + offset * sizeof(bits), &bits, sizeof(bits));
    }
}

}  // namespace

void SumCPU(const void* input, void* output, Dtype dtype, ShapeView shape,
            StrideView stride, std::size_t numel) {
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

void SumCPU(const void* input, void* output, Dtype dtype, ShapeView shape,
            StrideView stride, std::size_t numel, int dim) {
    const std::size_t count = numel / static_cast<std::size_t>(shape[dim]);
    const std::vector<int> reduction_shape{shape[dim]};
    const std::vector<std::size_t> reduction_stride{stride[dim]};
    const auto* input_bytes = static_cast<const std::byte*>(input);
    auto* output_bytes = static_cast<std::byte*>(output);
    const std::size_t input_size = dtype_size(dtype);
    const std::size_t output_size =
        (dtype == Dtype::kFloat32 || dtype == Dtype::kFloat16 || dtype == Dtype::kBFloat16)
            ? input_size : sizeof(std::int64_t);

    for (std::size_t i = 0; i < count; ++i) {
        std::size_t index = i;
        std::size_t offset = 0;
        // 从末维拆坐标，跳过归约维度，得到本组在输入中的起点。
        for (std::size_t axis = shape.size(); axis > 0; --axis) {
            if (axis - 1 == static_cast<std::size_t>(dim)) continue;
            const std::size_t length = static_cast<std::size_t>(shape[axis - 1]);
            offset += (index % length) * stride[axis - 1];
            index /= length;
        }
        // 每组是一个带 stride 的一维归约，复用已有 dtype 分发和累加规则。
        SumCPU(input_bytes + offset * input_size, output_bytes + i * output_size,
               dtype, reduction_shape, reduction_stride, reduction_shape[0]);
    }
}

void MeanCPU(const Tensor& input, int dim, bool keepdim, Tensor& out) {
    const Tensor total = dim == -1 ? input.sum() : input.sum(dim, keepdim);
    const float count = static_cast<float>(dim == -1 ? input.numel() : input.size(dim));
    const Tensor scale = full({}, 1.0f / count, input.dtype(), input.device());
    mul_out(total, scale.broadcast(out.shape()), out);
}

void MeanBackwardCPU(const Tensor& input, const Tensor& output_grad, int dim, Tensor& dx) {
    const float count = static_cast<float>(dim == -1 ? input.numel() : input.size(dim));
    Tensor scaled_grad = output_grad * (1.0f / count);
    if (dim >= 0 && scaled_grad.dim() != input.dim()) {
        scaled_grad = scaled_grad.unsqueeze(dim);
    }
    dx = scaled_grad.broadcast(input.shape());
}

void ExtremaCPU(const Tensor& input, Tensor& output, std::size_t inner_size, bool maximum) {
    const Dtype dtype = input.dtype();
    if (dtype == Dtype::kFloat32 || dtype == Dtype::kFloat16 || dtype == Dtype::kBFloat16) {
        ExtremaTypedCPU<float>(input, output, inner_size, maximum);
    } else {
        ExtremaTypedCPU<std::int64_t>(input, output, inner_size, maximum);
    }
}

void ExtremaGradCPU(const Tensor& input, const Tensor& extrema, const Tensor& grad,
                    Tensor& input_grad, std::size_t inner_size) {
    const Dtype dtype = input.dtype();
    const auto* data = input.data_ptr();
    auto* output = input_grad.data_ptr();
    const std::size_t groups = extrema.numel();
    const std::size_t count = input.numel() / groups;
    const bool contiguous = input.is_contiguous();
    for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t start = GroupStart(group, count, inner_size);
        const float best = ReadFloat(extrema.data_ptr(), dtype, group);
        const auto matches = [&](std::size_t i) {
            const auto offset = contiguous ? i : ElementOffset(i, input.shape(), input.stride());
            const float value = ReadFloat(data, dtype, offset);
            return value == best || (std::isnan(value) && std::isnan(best));
        };
        std::size_t ties = 0;
        for (std::size_t j = 0; j < count; ++j) {
            ties += matches(start + j * inner_size);
        }
        const float share = ReadFloat(grad.data_ptr(), dtype, group) / static_cast<float>(ties);
        for (std::size_t j = 0; j < count; ++j) {
            const std::size_t i = start + j * inner_size;
            WriteFloat(output, dtype, i, matches(i) ? share : 0.0f);
        }
    }
}

}  // namespace mtensor
