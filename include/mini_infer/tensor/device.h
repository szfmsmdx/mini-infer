#pragma once

#include <cstdint>

namespace mtensor {

enum class Device : std::uint8_t{
    kCPU,
    kCUDA
};

}  // namespace mtensor
