#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "ops/copy/kernels.h"
#include "mini_infer/tensor/tensor.h"
#include "ops/common/cpu_iteration.h"

namespace mtensor {

namespace {

float ReadFloat(const void* data, Dtype dtype, std::size_t offset) {
    switch (dtype) {
        case Dtype::kFloat16: return DecodeFloat16(static_cast<const std::uint16_t*>(data) + offset);
        case Dtype::kBFloat16: return DecodeBFloat16(static_cast<const std::uint16_t*>(data) + offset);
        case Dtype::kInt8: return static_cast<const std::int8_t*>(data)[offset];
        case Dtype::kUInt8: return static_cast<const std::uint8_t*>(data)[offset];
        case Dtype::kInt16: return static_cast<const std::int16_t*>(data)[offset];
        case Dtype::kInt32: return static_cast<float>(static_cast<const std::int32_t*>(data)[offset]);
        case Dtype::kInt64: return static_cast<float>(static_cast<const std::int64_t*>(data)[offset]);
        default: throw std::invalid_argument("CastCPU: unsupported input dtype");
    }
}

}  // namespace

void CopyCPU(
    void* dst_data_ptr,
    const void* src_data_ptr,
    ShapeView shape,
    StrideView dst_stride,
    StrideView src_stride,
    std::size_t element_size
) {
    auto* byte_dst_data_ptr = static_cast<std::byte*>(dst_data_ptr);
    auto* byte_src_data_ptr = static_cast<const std::byte*>(src_data_ptr);

    auto dfs = [&](
        auto&& self, std::size_t dim, std::size_t dst_offset, std::size_t src_offset
    ) -> void {
        if (dim == shape.size()) {
            std::memcpy(byte_dst_data_ptr + dst_offset, byte_src_data_ptr + src_offset, element_size);
            return;
        }

        for (int d = 0; d < shape[dim]; ++d) {
            const std::size_t next_dst_offset =
                dst_offset + static_cast<std::size_t>(d) * dst_stride[dim] * element_size;
            const std::size_t next_src_offset =
                src_offset + static_cast<std::size_t>(d) * src_stride[dim] * element_size;

            self(self, dim + 1, next_dst_offset, next_src_offset);
        }
    };

    dfs(dfs, 0, 0, 0);
}

void CastCPU(const Tensor& input, Tensor& output) {
    std::size_t i = 0;
    if (output.dtype() == Dtype::kFloat32) {
        const auto* src = input.data_ptr();
        auto* dst = static_cast<float*>(output.data_ptr());
        ForEachOffset(input.shape(), input.stride(), [&](std::size_t offset) {
            dst[i++] = ReadFloat(src, input.dtype(), offset);
        });
    } else {
        const auto* src = static_cast<const float*>(input.data_ptr());
        auto* dst = static_cast<std::uint16_t*>(output.data_ptr());
        const bool fp16 = output.dtype() == Dtype::kFloat16;
        ForEachOffset(input.shape(), input.stride(), [&](std::size_t offset) {
            dst[i++] = fp16 ? EncodeFloat16(src[offset]) : EncodeBFloat16(src[offset]);
        });
    }
}

}  // namespace mtensor
