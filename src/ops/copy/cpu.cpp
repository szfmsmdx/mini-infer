#include <cstddef>
#include <cstring>
#include <vector>

#include "ops/copy/kernels.h"

namespace mtensor {

void CopyCPU(
    void* dst_data_ptr,
    const void* src_data_ptr,
    const std::vector<int>& shape,
    const std::vector<std::size_t>& dst_stride,
    const std::vector<std::size_t>& src_stride,
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

}  // namespace mtensor
