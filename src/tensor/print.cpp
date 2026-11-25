#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor/print.h"
#include "mini_infer/tensor/tensor.h"
#include "memory/memory.h"
#include "tensor/check.h"

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
                 ShapeView shape, StrideView stride,
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

}  // namespace

void PrintCPU(const void* data_ptr, ShapeView shape,
              StrideView stride, Dtype dtype, bool is_empty) {
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

void Tensor::print() const {
    /*
    使用示例（a 为已构造的 CPU Tensor）：
        a.print();
    */
    const std::size_t count = numel();
    if (count > 0) CheckDataPointer(*this, "Tensor::print: null data pointer");

    if (device() == Device::kCPU) {
        PrintCPU(data_ptr(), shape(), stride(), dtype(), empty());
    } else if (device() == Device::kCUDA) {
#ifdef ENABLE_CUDA
        if (!is_contiguous()) {
            clone().print();
            return;
        }
        const std::size_t bytes = nbytes();
        std::vector<std::byte> host_buffer(bytes);
        if (bytes > 0) {
            CopyBytes(host_buffer.data(), Device::kCPU, data_ptr(), device(), bytes);
        }
        PrintCPU(host_buffer.data(), shape(), stride(), dtype(), empty());
#else
        throw std::logic_error("Tensor::print: CUDA support is disabled");
#endif
    } else {
        throw std::invalid_argument("Tensor::print: unsupported device");
    }
}

}  // namespace mtensor
