#pragma once

#include <cstddef>
#include <vector>

namespace mtensor {

// 按逻辑顺序访问元素，回调接收以元素为单位的存储偏移。
template <typename Visit>
void ForEachOffset(const std::vector<int>& shape,
                   const std::vector<std::size_t>& stride, Visit visit) {
    auto walk = [&](auto&& self, std::size_t dim, std::size_t offset) -> void {
        if (dim == shape.size()) {
            visit(offset);
            return;
        }
        for (int i = 0; i < shape[dim]; ++i) {
            self(self, dim + 1, offset + static_cast<std::size_t>(i) * stride[dim]);
        }
    };
    walk(walk, 0, 0);
}

template <typename Visit>
void ForEachOffset(const std::vector<int>& shape,
                   const std::vector<std::size_t>& a_stride,
                   const std::vector<std::size_t>& b_stride, Visit visit) {
    auto walk = [&](auto&& self, std::size_t dim,
                    std::size_t a_offset, std::size_t b_offset) -> void {
        if (dim == shape.size()) {
            visit(a_offset, b_offset);
            return;
        }
        for (int i = 0; i < shape[dim]; ++i) {
            const auto index = static_cast<std::size_t>(i);
            self(self, dim + 1, a_offset + index * a_stride[dim],
                 b_offset + index * b_stride[dim]);
        }
    };
    walk(walk, 0, 0, 0);
}

template <typename Visit>
void ForEachOffset(const std::vector<int>& shape,
                   const std::vector<std::size_t>& a_stride,
                   const std::vector<std::size_t>& b_stride,
                   const std::vector<std::size_t>& output_stride, Visit visit) {
    auto walk = [&](auto&& self, std::size_t dim, std::size_t a_offset,
                    std::size_t b_offset, std::size_t output_offset) -> void {
        if (dim == shape.size()) {
            visit(a_offset, b_offset, output_offset);
            return;
        }
        for (int i = 0; i < shape[dim]; ++i) {
            const auto index = static_cast<std::size_t>(i);
            self(self, dim + 1, a_offset + index * a_stride[dim],
                 b_offset + index * b_stride[dim],
                 output_offset + index * output_stride[dim]);
        }
    };
    walk(walk, 0, 0, 0, 0);
}

}  // namespace mtensor
