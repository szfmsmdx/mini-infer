#include "utils/memory.h"

#include <cstdlib>
#include <new>
#include <string>

#include "utils/logger.h"

namespace mtensor {

void* AllocateCPU(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }

    void* ptr = std::malloc(bytes);
    if (ptr == nullptr) {
        Logger::error("Tensor allocation failed: " + std::to_string(bytes) + " bytes");
        throw std::bad_alloc{};
    }
    return ptr;
}

}  // namespace mtensor
