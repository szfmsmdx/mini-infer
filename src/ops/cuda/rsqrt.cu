#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/backend.h"
#include "mini_infer/tensor.h"
#include "ops/cuda/index.cuh"
#include "utils/memory.h"

namespace mtensor {
namespace {

// 连续输入直接用 i；视图按各自的 stride 定位。输出总是连续。
template <typename In, typename Out>
__global__ void RsqrtKernel(const In* x, Out* out, std::size_t count, const int* shape,
                            const std::size_t* xs, int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = xs ? GetNumelIndex(i, shape, xs, ndim) : i;
    const float y = rsqrtf(static_cast<float>(x[xi]));
    out[i] = static_cast<Out>(y);
}

template <typename In, typename Out>
void LaunchRsqrt(const Tensor& x, Tensor& out) {
    const auto n = x.numel();
    if (n == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n - 1) / threads + 1);
    RsqrtKernel<In, Out><<<blocks, threads>>>(
        static_cast<const In*>(x.data_ptr()), static_cast<Out*>(out.data_ptr()), n, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void RsqrtCUDA(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchRsqrt<float, float>(x, out); break;
        case Dtype::kFloat16: LaunchRsqrt<__half, __half>(x, out); break;
        case Dtype::kBFloat16: LaunchRsqrt<__nv_bfloat16, __nv_bfloat16>(x, out); break;
        case Dtype::kInt8: LaunchRsqrt<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: LaunchRsqrt<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: LaunchRsqrt<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: LaunchRsqrt<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: LaunchRsqrt<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("rsqrt: unsupported dtype");
    }
}

}  // namespace mtensor
