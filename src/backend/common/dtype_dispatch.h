#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "mini_infer/tensor/dtype.h"

namespace mtensor {

template <typename T>
struct TypeTag {
    using type = T;
};

// The callback receives a type tag; each backend chooses how to use that type.
template <typename Fn>
decltype(auto) DispatchFloatingDtypes(Dtype dtype, Fn&& fn) {
    switch (dtype) {
        case Dtype::kFloat32: return std::forward<Fn>(fn)(TypeTag<float>{});
        case Dtype::kFloat16: return std::forward<Fn>(fn)(TypeTag<Float16>{});
        case Dtype::kBFloat16: return std::forward<Fn>(fn)(TypeTag<BFloat16>{});
        default: throw std::invalid_argument("floating dtype dispatch: unsupported dtype");
    }
}

template <typename Fn>
decltype(auto) DispatchIntegralDtypes(Dtype dtype, Fn&& fn) {
    switch (dtype) {
        case Dtype::kInt8: return std::forward<Fn>(fn)(TypeTag<std::int8_t>{});
        case Dtype::kUInt8: return std::forward<Fn>(fn)(TypeTag<std::uint8_t>{});
        case Dtype::kInt16: return std::forward<Fn>(fn)(TypeTag<std::int16_t>{});
        case Dtype::kInt32: return std::forward<Fn>(fn)(TypeTag<std::int32_t>{});
        case Dtype::kInt64: return std::forward<Fn>(fn)(TypeTag<std::int64_t>{});
        default: throw std::invalid_argument("integral dtype dispatch: unsupported dtype");
    }
}

template <typename Fn>
decltype(auto) DispatchAllDtypes(Dtype dtype, Fn&& fn) {
    switch (dtype) {
        case Dtype::kFloat32: return std::forward<Fn>(fn)(TypeTag<float>{});
        case Dtype::kFloat16: return std::forward<Fn>(fn)(TypeTag<Float16>{});
        case Dtype::kBFloat16: return std::forward<Fn>(fn)(TypeTag<BFloat16>{});
        case Dtype::kInt8: return std::forward<Fn>(fn)(TypeTag<std::int8_t>{});
        case Dtype::kUInt8: return std::forward<Fn>(fn)(TypeTag<std::uint8_t>{});
        case Dtype::kInt16: return std::forward<Fn>(fn)(TypeTag<std::int16_t>{});
        case Dtype::kInt32: return std::forward<Fn>(fn)(TypeTag<std::int32_t>{});
        case Dtype::kInt64: return std::forward<Fn>(fn)(TypeTag<std::int64_t>{});
        default: throw std::invalid_argument("dtype dispatch: unsupported dtype");
    }
}

}  // namespace mtensor
