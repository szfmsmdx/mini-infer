#include "backend/dispatch.h"

namespace mtensor {

const KernelTable& GetKernelTable() {
    static const KernelTable table = [] {
        KernelTable result;
        RegisterCPUKernels(result);
#ifdef ENABLE_CUDA
        RegisterCUDAKernels(result);
#endif
        return result;
    }();
    return table;
}

}  // namespace mtensor
