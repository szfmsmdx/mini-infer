#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor/tensor.h"
#include "tensor/dtype.h"

namespace mtensor {
namespace {

void PrintElement(std::ostream& out, Dtype dtype, const void* element_ptr) {
    switch (dtype) {
        case Dtype::kFloat32:
            out << *static_cast<const float*>(element_ptr);
            return;
        case Dtype::kFloat16:
            out << DecodeFloat16(element_ptr);
            return;
        case Dtype::kBFloat16:
            out << DecodeBFloat16(element_ptr);
            return;
        case Dtype::kInt8:
            out << static_cast<int>(*static_cast<const std::int8_t*>(element_ptr));
            return;
        case Dtype::kUInt8:
            out << static_cast<unsigned int>(*static_cast<const std::uint8_t*>(element_ptr));
            return;
        case Dtype::kInt16:
            out << *static_cast<const std::int16_t*>(element_ptr);
            return;
        case Dtype::kInt32:
            out << *static_cast<const std::int32_t*>(element_ptr);
            return;
        case Dtype::kInt64:
            out << *static_cast<const std::int64_t*>(element_ptr);
            return;
    }
    throw std::invalid_argument("Tensor::print: unsupported dtype");
}

void PrintNested(std::ostream& out, Dtype dtype, const std::byte* data_bytes_ptr,
                 const std::vector<int>& shape, const std::vector<std::size_t>& stride,
                 std::size_t dimension, std::size_t element_bytes, std::size_t offset) {
    out << '[';
    for (int index = 0; index < shape[dimension]; ++index) {
        if (index > 0) {
            out << ',';
            if (dimension + 1 == shape.size()) {
                out << ' ';
            } else {
                // Align the next slice under the opening '[' after "tensor(".
                out << '\n' << std::string(8 + dimension, ' ');
            }
        }

        const std::size_t next_offset = offset +
            static_cast<std::size_t>(index) * stride[dimension] * element_bytes;
        if (dimension + 1 == shape.size()) {
            PrintElement(out, dtype, data_bytes_ptr + next_offset);
        } else {
            PrintNested(out, dtype, data_bytes_ptr, shape, stride, dimension + 1,
                        element_bytes, next_offset);
        }
    }
    out << ']';
}

}  // namespace, 这里匿名包住只供本文件使用

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

void ZeroCPU(
    void* data_ptr,
    const std::vector<int>& shape,
    const std::vector<std::size_t>& stride,
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
    const std::vector<int>& shape,
    const std::vector<std::size_t>& stride,
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



void PrintCPU(const void* data_ptr, const std::vector<int>& shape,
              const std::vector<std::size_t>& stride, Dtype dtype, bool is_empty) {
    std::cout << "tensor(";
    if (shape.empty()) {
        if (is_empty) {
            std::cout << "[]";
        } else {
            PrintElement(std::cout, dtype, data_ptr);
        }
    } else {
        const auto* data_bytes_ptr = static_cast<const std::byte*>(data_ptr);
        PrintNested(std::cout, dtype, data_bytes_ptr, shape, stride, 0, dtype_size(dtype), 0);
    }
    std::cout << ")" << std::endl;
}

}  // namespace mtensor
