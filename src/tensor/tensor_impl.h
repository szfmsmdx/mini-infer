#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "mini_infer/tensor/dtype.h"
#include "mini_infer/tensor/layout_view.h"
#include "mini_infer/tensor/storage.h"

namespace mtensor {

struct AutogradMetadata;

void CheckLayoutRank(std::size_t rank);

struct TensorImpl {
    TensorImpl(std::shared_ptr<Storage> storage, ShapeView shape, StrideView stride,
               std::size_t storage_offset, Dtype dtype, Device device,
               std::shared_ptr<std::uint64_t> version = {});

    std::shared_ptr<Storage> storage;
    std::size_t storage_offset;  // 相对 Storage 起点的元素偏移，不是相对父视图
    Dtype dtype;
    Device empty_device;  // 仅在没有 Storage 时使用；其余情况以 Storage 的设备为准
    int rank;
    int shape[kMaxDim]{};
    std::size_t stride[kMaxDim]{};
    std::size_t numel;
    std::shared_ptr<std::uint64_t> version;
    std::shared_ptr<AutogradMetadata> autograd_meta;
};

}  // namespace mtensor
