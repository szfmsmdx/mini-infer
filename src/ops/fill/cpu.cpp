#include <cstddef>
#include <cstring>

#include "ops/fill/kernels.h"

namespace mtensor {

void ZeroCPU(
    void* data_ptr,
    ShapeView shape,
    StrideView stride,
    std::size_t element_size
) {
    auto* byte_data_ptr = static_cast<std::byte*>(data_ptr);

    // offset 的单位是字节。
    auto dfs = [&](auto&& self, std::size_t dim, std::size_t offset) -> void {
        if (dim == shape.size()) {
            std::memset(byte_data_ptr + offset, 0, element_size);
            return;
        }

        for (int d = 0; d < shape[dim]; ++d) {
            const std::size_t next_offset =
                offset + static_cast<std::size_t>(d) * stride[dim] * element_size;

            // self 是当前 lambda；把它自身传入下一层，才能在 C++17 中递归调用。
            self(self, dim + 1, next_offset);
        }
    };

    // 第一个参数传入 dfs 自身；从第 0 维、字节偏移 0 开始遍历。
    dfs(dfs, 0, 0);
}

void FillCPU(
    void* data_ptr,
    ShapeView shape,
    StrideView stride,
    std::size_t element_size,
    const void* host_value
) {
    auto* byte_data_ptr = static_cast<std::byte*>(data_ptr);
    auto dfs = [&](auto&& self, std::size_t dim, std::size_t offset) -> void {
        if (dim == shape.size()) {
            std::memcpy(byte_data_ptr + offset, host_value, element_size);
            return;
        }

        for (int d = 0; d < shape[dim]; ++d) {
            const std::size_t next_offset =
                offset + static_cast<std::size_t>(d) * stride[dim] * element_size;
            self(self, dim + 1, next_offset);
        }
    };
    dfs(dfs, 0, 0);
}

}  // namespace mtensor
