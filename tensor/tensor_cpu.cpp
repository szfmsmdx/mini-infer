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

void PrintElement(std::ostream& out, Dtype dtype, const void* element) {
    switch (dtype) {
        case Dtype::kFloat32:
            out << *static_cast<const float*>(element);
            return;
        case Dtype::kFloat16:
            out << DecodeFloat16(element);
            return;
        case Dtype::kBFloat16:
            out << DecodeBFloat16(element);
            return;
        case Dtype::kInt8:
            out << static_cast<int>(*static_cast<const std::int8_t*>(element));
            return;
        case Dtype::kUInt8:
            out << static_cast<unsigned int>(*static_cast<const std::uint8_t*>(element));
            return;
        case Dtype::kInt16:
            out << *static_cast<const std::int16_t*>(element);
            return;
        case Dtype::kInt32:
            out << *static_cast<const std::int32_t*>(element);
            return;
        case Dtype::kInt64:
            out << *static_cast<const std::int64_t*>(element);
            return;
    }
    throw std::invalid_argument("Tensor::print: unsupported dtype");
}

void PrintNested(std::ostream& out, Dtype dtype, const std::byte* data_bytes_ptr,
                 const std::vector<int>& shape, std::size_t dimension,
                 std::size_t element_bytes, std::size_t& flat_index) {
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

        if (dimension + 1 == shape.size()) {
            PrintElement(out, dtype, data_bytes_ptr + flat_index * element_bytes);
            ++flat_index;
        } else {
            PrintNested(out, dtype, data_bytes_ptr, shape, dimension + 1,
                        element_bytes, flat_index);
        }
    }
    out << ']';
}

}  // namespace

void CopyCPU(void* destination, const void* source, std::size_t bytes) {
    std::memcpy(destination, source, bytes);
}

void ZeroCPU(void* data, std::size_t bytes) {
    std::memset(data, 0, bytes);
}

void FillCPU(void* data, std::size_t count, Dtype dtype, const void* host_value) {
    if (count == 0) {
        return;
    }
    if (dtype == Dtype::kFloat16 || dtype == Dtype::kBFloat16) {
        const auto value = *static_cast<const std::uint16_t*>(host_value);
        auto* ptr = static_cast<std::uint16_t*>(data);
        for (std::size_t i = 0; i < count; ++i) {
            ptr[i] = value;
        }
        return;
    }
    DISPATCH_DTYPE(dtype, Element, {
        const Element value = *static_cast<const Element*>(host_value);
        Element* ptr = static_cast<Element*>(data);
        for (std::size_t i = 0; i < count; ++i) {
            ptr[i] = value;
        }
    });
}

void PrintCPU(const void* data, const std::vector<int>& shape, Dtype dtype, bool is_empty) {
    std::cout << "tensor(";
    if (shape.empty()) {
        if (is_empty) {
            std::cout << "[]";
        } else {
            PrintElement(std::cout, dtype, data);
        }
    } else {
        const auto* data_bytes_ptr = static_cast<const std::byte*>(data);
        std::size_t flat_index = 0;
        PrintNested(std::cout, dtype, data_bytes_ptr, shape, 0, dtype_size(dtype), flat_index);
    }
    std::cout << ")" << std::endl;
}

}  // namespace mtensor
