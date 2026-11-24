#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "mini_infer/core/dtype.h"
#include "mini_infer/core/layout_view.h"
#include "mini_infer/core/storage.h"

namespace mtensor {

struct AutogradMetadata;

void CheckLayoutRank(std::size_t rank);

struct TensorImpl {
    TensorImpl(std::shared_ptr<Storage> storage, ShapeView shape, StrideView stride,
               std::size_t storage_offset, Dtype dtype, Device device,
               std::shared_ptr<std::uint64_t> version = {});

    std::shared_ptr<Storage> storage;
    std::size_t storage_offset;  // elements from Storage base, not from a parent view
    Dtype dtype;
    Device empty_device;  // device for storage-less Tensors
    int rank;
    int shape[kMaxDim]{};
    std::size_t stride[kMaxDim]{};
    std::size_t numel;
    std::shared_ptr<std::uint64_t> version;
    std::shared_ptr<AutogradMetadata> autograd_meta;
};

}  // namespace mtensor
