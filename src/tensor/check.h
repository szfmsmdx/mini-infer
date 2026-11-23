#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mini_infer/dtype.h"

namespace mtensor {
class Tensor;

// 基础检查保留调用方的错误信息，失败抛 invalid_argument。
void CheckNotEmpty(const Tensor& tensor, const char* message);
void CheckSameShape(const Tensor& a, const Tensor& b, const char* message);
void CheckSameDtype(const Tensor& a, const Tensor& b, const char* message);
void CheckSameDevice(const Tensor& a, const Tensor& b, const char* message);
void CheckFloatingDtype(Dtype dtype, const char* message);
void CheckNonnegativeShape(const std::vector<int>& shape, const char* message);

// 指针检查抛 logic_error；维度必须在 [0, ndim) 内，否则抛 out_of_range。
void CheckDataPointer(const Tensor& tensor, const char* message);
void CheckVersion(const Tensor& tensor, std::uint64_t expected, const char* message);
void CheckDim(int dim, std::size_t ndim, const char* message);
void CheckBroadcastWritable(const Tensor& tensor);
void CheckPermutation(const std::vector<int>& dims, std::size_t ndim);
void CheckBroadcastTarget(const std::vector<int>& shape, const std::vector<int>& target_shape);

}  // namespace mtensor
