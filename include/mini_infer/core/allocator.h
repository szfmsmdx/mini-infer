#pragma once

#include <cstddef>
#include <memory>

#include "mini_infer/core/device.h"

namespace mtensor {

using DataDeleter = void (*)(void*) noexcept;   // 函数指针

// DataPtr owns one allocation; moving transfers its device and deleter.
class DataPtr {
public:
    DataPtr(void* data, Device device, DataDeleter deleter) noexcept
        : data_(data, deleter), device_(device) {}

    void* get() const noexcept { return data_.get(); }
    Device device() const noexcept { return device_; }

private:
    std::unique_ptr<void, DataDeleter> data_;
    Device device_;
};

class Allocator {
public:
    virtual ~Allocator() = default;
    virtual DataPtr allocate(std::size_t nbytes) const = 0;
};

}  // namespace mtensor
