#pragma once

#include <cstddef>
#include <utility>

#include "mini_infer/tensor/allocator.h"

namespace mtensor {

class Storage {
public:
    Storage(DataPtr data, std::size_t nbytes) noexcept
        : data_(std::move(data)), nbytes_(nbytes) {}

    void* data() const noexcept { return data_.get(); }
    std::size_t nbytes() const noexcept { return nbytes_; }
    Device device() const noexcept { return data_.device(); }

private:
    DataPtr data_;
    std::size_t nbytes_;
};

}  // namespace mtensor
