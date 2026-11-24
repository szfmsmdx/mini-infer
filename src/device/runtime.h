#pragma once

#include <cstddef>

#include "mini_infer/core/allocator.h"

namespace mtensor {

const Allocator& GetAllocator(Device device);
void CopyBytes(void* dst, Device dst_device, const void* src, Device src_device,
               std::size_t bytes);

}  // namespace mtensor
