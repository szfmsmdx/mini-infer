#include <stdexcept>

#include "tensor/check.h"
#include "mini_infer/core/tensor.h"

namespace mtensor {

void CheckNotEmpty(const Tensor& tensor, const char* message) {
    if (tensor.empty()) throw std::invalid_argument(message);
}

void CheckSameShape(const Tensor& a, const Tensor& b, const char* message) {
    if (a.shape() != b.shape()) throw std::invalid_argument(message);
}

void CheckSameDtype(const Tensor& a, const Tensor& b, const char* message) {
    if (a.dtype() != b.dtype()) throw std::invalid_argument(message);
}

void CheckSameDevice(const Tensor& a, const Tensor& b, const char* message) {
    if (a.device() != b.device()) throw std::invalid_argument(message);
}

void CheckFloatingDtype(Dtype dtype, const char* message) {
    if (dtype != Dtype::kFloat32 && dtype != Dtype::kFloat16 && dtype != Dtype::kBFloat16)
        throw std::invalid_argument(message);
}

void CheckDataPointer(const Tensor& tensor, const char* message) {
    if (!tensor.data_ptr()) throw std::logic_error(message);
}

void CheckVersion(const Tensor& tensor, std::uint64_t expected, const char* message) {
    if (tensor.version() != expected) throw std::logic_error(message);
}

void CheckDim(int dim, std::size_t ndim, const char* message) {
    if (dim < 0 || dim >= ndim) throw std::out_of_range(message);
}


void CheckBroadcastWritable(const Tensor& tensor) {
    if (tensor.numel() == 0) return;
    for (std::size_t i = 0; i < tensor.shape().size(); ++i) {
        if (tensor.shape()[i] > 1 && tensor.stride()[i] == 0) {
            throw std::invalid_argument("Cannot write to a broadcast view with overlapping elements");
        }
    }
}

void CheckPermutation(const std::vector<int>& dims, std::size_t ndim) {
    if (dims.size() != ndim) {
        throw std::invalid_argument("Tensor::permute: dimension count must match tensor rank");
    }
    std::vector<bool> seen(ndim, false);
    for (int dim : dims) {
        CheckDim(dim, ndim, "Tensor::permute: dimension index out of range");
        if (seen[dim]) {
            throw std::invalid_argument("Tensor::permute: duplicate dimension index");
        }
        seen[dim] = true;
    }
}

void CheckBroadcastTarget(ShapeView shape, ShapeView target_shape) {
    if (shape.size() > target_shape.size()) {
        throw std::invalid_argument("broadcast: target rank is too small");
    }
    for (int size : target_shape) {
        if (size <= 0) {
            throw std::invalid_argument("broadcast: target dimensions must be positive");
        }
    }
    const std::size_t leading = target_shape.size() - shape.size();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != target_shape[leading + i] && shape[i] != 1) {
            throw std::invalid_argument("broadcast: incompatible shape");
        }
    }

}

}  // namespace mtensor
