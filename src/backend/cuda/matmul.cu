#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/matmul/kernels.h"
#include "mini_infer/ops/matmul.h"
#include "runtime/cuda_check.h"

namespace mtensor {
namespace {

struct MatrixStride {
    std::size_t row;
    std::size_t col;
};

constexpr int kThreads = 256;

__device__ float ToFloat(__half value) { return __half2float(value); }
__device__ float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }
__device__ void StoreHalf(__half& output, float value) { output = __float2half_rn(value); }
__device__ void StoreHalf(__nv_bfloat16& output, float value) {
    output = __float2bfloat16_rn(value);
}

__global__ void SgemmKernel(const float* a, const float* b, float* output,
                            int k, int n, std::size_t numel,
                            MatrixStride a_stride, MatrixStride b_stride) {
    for (std::size_t index = threadIdx.x; index < numel; index += blockDim.x) {
        const std::size_t a_row = (index / n) * a_stride.row;
        const std::size_t b_col = (index % n) * b_stride.col;
        float sum = 0;
        for (int p = 0; p < k; ++p) {
            sum += a[a_row + p * a_stride.col] * b[p * b_stride.row + b_col];
        }
        output[index] = sum;
    }
}

template <typename Half>
__global__ void MatmulHalfKernel(const Half* a, const Half* b, Half* output,
                                 int k, int n, std::size_t numel,
                                 MatrixStride a_stride, MatrixStride b_stride) {
    for (std::size_t index = threadIdx.x; index < numel; index += blockDim.x) {
        const std::size_t a_row = (index / n) * a_stride.row;
        const std::size_t b_col = (index % n) * b_stride.col;
        float sum = 0;
        for (int p = 0; p < k; ++p) {
            sum += ToFloat(a[a_row + p * a_stride.col]) *
                   ToFloat(b[p * b_stride.row + b_col]);
        }
        StoreHalf(output[index], sum);
    }
}

template <typename U>
__global__ void MatmulIntegerKernel(const void* a, const void* b, void* output,
                                    int k, int n, std::size_t numel,
                                    MatrixStride a_stride, MatrixStride b_stride) {
    using Acc = std::conditional_t<(sizeof(U) < sizeof(std::uint32_t)), std::uint32_t, U>;
    const auto* a_data = static_cast<const U*>(a);
    const auto* b_data = static_cast<const U*>(b);
    auto* output_data = static_cast<U*>(output);
    for (std::size_t index = threadIdx.x; index < numel; index += blockDim.x) {
        const std::size_t a_row = (index / n) * a_stride.row;
        const std::size_t b_col = (index % n) * b_stride.col;
        Acc sum = 0;
        for (int p = 0; p < k; ++p) {
            const Acc lhs = a_data[a_row + p * a_stride.col];
            const Acc rhs = b_data[p * b_stride.row + b_col];
            sum += lhs * rhs;
        }
        output_data[index] = static_cast<U>(sum);
    }
}

}  // namespace

void MatmulCUDA(const void* a, const void* b, void* output, Dtype dtype,
                int m, int k, int n, StrideView a_stride,
                StrideView b_stride) {
    if (m == 0 || n == 0) return;
    const std::size_t numel = static_cast<std::size_t>(m) * n;
    const MatrixStride lhs_stride{a_stride[0], a_stride[1]};
    const MatrixStride rhs_stride{b_stride[0], b_stride[1]};
    switch (dtype) {
        case Dtype::kFloat32:
            SgemmKernel<<<1, kThreads>>>(static_cast<const float*>(a),
                                        static_cast<const float*>(b),
                                        static_cast<float*>(output),
                                        k, n, numel, lhs_stride, rhs_stride);
            break;
        case Dtype::kFloat16:
            MatmulHalfKernel<<<1, kThreads>>>(static_cast<const __half*>(a),
                                             static_cast<const __half*>(b),
                                             static_cast<__half*>(output),
                                             k, n, numel, lhs_stride, rhs_stride);
            break;
        case Dtype::kBFloat16:
            MatmulHalfKernel<<<1, kThreads>>>(static_cast<const __nv_bfloat16*>(a),
                                             static_cast<const __nv_bfloat16*>(b),
                                             static_cast<__nv_bfloat16*>(output),
                                             k, n, numel, lhs_stride, rhs_stride);
            break;
        // 用无符号运算保留输出类型的回绕语义。
        case Dtype::kInt8:
        case Dtype::kUInt8:
            MatmulIntegerKernel<std::uint8_t><<<1, kThreads>>>(a, b, output, k, n, numel, lhs_stride, rhs_stride);
            break;
        case Dtype::kInt16:
            MatmulIntegerKernel<std::uint16_t><<<1, kThreads>>>(a, b, output, k, n, numel, lhs_stride, rhs_stride);
            break;
        case Dtype::kInt32:
            MatmulIntegerKernel<std::uint32_t><<<1, kThreads>>>(a, b, output, k, n, numel, lhs_stride, rhs_stride);
            break;
        case Dtype::kInt64:
            MatmulIntegerKernel<std::uint64_t><<<1, kThreads>>>(a, b, output, k, n, numel, lhs_stride, rhs_stride);
            break;
        default:
            throw std::invalid_argument("MatmulCUDA: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

void MatmulBackwardCUDA(const Tensor& a, const Tensor& b, const Tensor& grad,
                        Tensor& da, Tensor& db) {
    matmul_out(grad, b.transpose(0, 1), da);
    matmul_out(a.transpose(0, 1), grad, db);
}

}  // namespace mtensor
