#include <stdexcept>

#include "ops/matmul.h"
#include "ops/backend.h"
#include "tensor/check.h"

namespace mtensor {

Tensor matmul(const Tensor& a, const Tensor& b) {
    CheckMatmul(a, b);

    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);
    if (a.device() == Device::kCPU) {
        Tensor output({m, n}, a.dtype(), a.device());
        MatmulCPU(a.data_ptr(), b.data_ptr(), output.data_ptr(), a.dtype(),
                  m, k, n, a.stride(), b.stride());
        return output;
    }
#ifdef ENABLE_CUDA
    if (a.device() == Device::kCUDA) {
        Tensor output({m, n}, a.dtype(), a.device());
        MatmulCUDA(a.data_ptr(), b.data_ptr(), output.data_ptr(), a.dtype(),
                   m, k, n, a.stride(), b.stride());
        return output;
    }
#endif
    throw std::logic_error("matmul: no backend implemented for this device");
}

Tensor Tensor::matmul(const Tensor& other) const {
    return mtensor::matmul(*this, other);
}

}  // namespace mtensor
