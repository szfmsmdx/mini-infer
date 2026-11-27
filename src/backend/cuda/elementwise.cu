#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ops/elementwise/kernels.h"
#include "mini_infer/tensor/tensor.h"
#include "backend/common/cuda_index.cuh"
#include "backend/common/cuda_unary.cuh"
#include "runtime/cuda_check.h"

namespace mtensor {
namespace {
enum class Op { Add, Sub, Mul, Div, Neg };

template <typename T, Op op>
__device__ T Calculate(T a, T b) {
    if constexpr (std::is_integral_v<T>) {  // int 回环机制
        using U = std::make_unsigned_t<T>;
        if constexpr (op == Op::Div) {
            if (b == 0) return 0;
            if constexpr (std::is_signed_v<T>) {
                const U min_bits = U{1} << (sizeof(T) * 8 - 1);
                if (static_cast<U>(a) == min_bits && b == T(-1)) return a;
            }
            return static_cast<T>(a / b);
        } else {
            const U ua = static_cast<U>(a);
            U bits;
            if constexpr (op == Op::Neg) {
                bits = static_cast<U>(U{0} - ua);
            } else {
                const U ub = static_cast<U>(b);
                if constexpr (op == Op::Add) bits = static_cast<U>(ua + ub);
                if constexpr (op == Op::Sub) bits = static_cast<U>(ua - ub);
                if constexpr (op == Op::Mul) {
                    const auto product = static_cast<unsigned long long>(ua) * ub;
                    bits = static_cast<U>(product);
                }
            }
            T result;
            memcpy(&result, &bits, sizeof(T));
            return result;
        }
    } else {
        if constexpr (op == Op::Add) return a + b;
        if constexpr (op == Op::Sub) return a - b;
        if constexpr (op == Op::Mul) return a * b;
        if constexpr (op == Op::Div) return a / b;
        if constexpr (op == Op::Neg) return -a;
    }
}

template <typename T, Op op>
__global__ void ElementwiseKernel(const T* a, const T* b, T* out, int numel,
                                  CUDALayout a_layout, CUDALayout b_layout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    const T av = a[GetNumelIndex(idx, a_layout)];
    T bv{};
    if constexpr (op != Op::Neg) bv = b[GetNumelIndex(idx, b_layout)];
    out[idx] = Calculate<T, op>(av, bv);
}

template <typename Half, Op op>
__global__ void HalfKernel(const std::uint16_t* a, const std::uint16_t* b,
                           std::uint16_t* out, int numel, CUDALayout a_layout, CUDALayout b_layout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    const std::uint16_t abits = a[GetNumelIndex(idx, a_layout)];
    if constexpr (op == Op::Neg) {
        out[idx] = abits ^ 0x8000u;
    } else {
        const std::uint16_t bbits = b[GetNumelIndex(idx, b_layout)];
        float av, bv;
        if constexpr (std::is_same_v<Half, __half>) {
            av = __half2float(__ushort_as_half(abits));
            bv = __half2float(__ushort_as_half(bbits));
        } else {
            av = __bfloat162float(__ushort_as_bfloat16(abits));
            bv = __bfloat162float(__ushort_as_bfloat16(bbits));
        }
        const float value = Calculate<float, op>(av, bv);
        if constexpr (std::is_same_v<Half, __half>) out[idx] = __half_as_ushort(__float2half_rn(value));
        else out[idx] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
}

template <typename T, Op op>
void LaunchTyped(const void* a, const void* b, void* output, std::size_t numel,
                 CUDALayout a_layout, CUDALayout b_layout, unsigned blocks, unsigned threads) {
    ElementwiseKernel<T, op><<<blocks, threads>>>(static_cast<const T*>(a),
        static_cast<const T*>(b), static_cast<T*>(output), numel, a_layout, b_layout);
}

template <Op op>
void Launch(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
            CUDALayout a_layout, CUDALayout b_layout) {
    if (numel == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((numel - 1) / threads + 1);
    switch (dtype) {
        case Dtype::kFloat32: LaunchTyped<float, op>(a,b,output,numel,a_layout, b_layout,blocks,threads); break;
        case Dtype::kInt8: LaunchTyped<std::int8_t, op>(a,b,output,numel,a_layout, b_layout,blocks,threads); break;
        case Dtype::kUInt8: LaunchTyped<std::uint8_t, op>(a,b,output,numel,a_layout, b_layout,blocks,threads); break;
        case Dtype::kInt16: LaunchTyped<std::int16_t, op>(a,b,output,numel,a_layout, b_layout,blocks,threads); break;
        case Dtype::kInt32: LaunchTyped<std::int32_t, op>(a,b,output,numel,a_layout, b_layout,blocks,threads); break;
        case Dtype::kInt64: LaunchTyped<std::int64_t, op>(a,b,output,numel,a_layout, b_layout,blocks,threads); break;
        case Dtype::kFloat16: HalfKernel<__half, op><<<blocks, threads>>>(static_cast<const std::uint16_t*>(a), static_cast<const std::uint16_t*>(b), static_cast<std::uint16_t*>(output), numel, a_layout, b_layout); break;
        case Dtype::kBFloat16: HalfKernel<__nv_bfloat16, op><<<blocks, threads>>>(static_cast<const std::uint16_t*>(a), static_cast<const std::uint16_t*>(b), static_cast<std::uint16_t*>(output), numel, a_layout, b_layout); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Input, typename Output>
__global__ void SqrtKernel(const Input* input, Output* output, std::size_t numel,
                           CUDALayout layout) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float x = static_cast<float>(input[GetNumelIndex(i, layout)]);
    output[i] = static_cast<Output>(sqrtf(x));
}

template <typename Input, typename Output>
void LaunchSqrt(const void* input, void* output, std::size_t numel,
                CUDALayout layout) {
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((numel - 1) / threads + 1);
    SqrtKernel<<<blocks, threads>>>(static_cast<const Input*>(input), static_cast<Output*>(output),
                                    numel, layout);
}

// 连续输入直接用 i；视图按各自的 stride 定位。输出总是连续。
template <typename In, typename Out>
__global__ void RsqrtKernel(const In* x, Out* out, std::size_t count, CUDALayout x_layout) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = GetNumelIndex(i, x_layout);
    const float y = rsqrtf(static_cast<float>(x[xi]));
    out[i] = static_cast<Out>(y);
}

template <typename In, typename Out>
void LaunchRsqrt(const Tensor& x, Tensor& out) {
    LaunchUnary<In, Out, RsqrtKernel<In, Out>>(x, out);
}

// 连续输入直接用 i；视图按各自的 stride 定位。输出总是连续。
template <typename In, typename Out, bool Log>
__global__ void ExpLogKernel(const In* x, Out* out, std::size_t count, CUDALayout x_layout) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = GetNumelIndex(i, x_layout);
    const float v = static_cast<float>(x[xi]);
    const float y = Log ? logf(v) : expf(v);
    out[i] = static_cast<Out>(y);
}

template <typename In, typename Out, bool Log>
void LaunchExpLog(const Tensor& x, Tensor& out) {
    LaunchUnary<In, Out, ExpLogKernel<In, Out, Log>>(x, out);
}


}  // namespace

void AddCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout, CUDALayout b_layout) {
    Launch<Op::Add>(a, b, output, numel, dtype, a_layout, b_layout);
}
void SubCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout, CUDALayout b_layout) {
    Launch<Op::Sub>(a, b, output, numel, dtype, a_layout, b_layout);
}
void MulCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout, CUDALayout b_layout) {
    Launch<Op::Mul>(a, b, output, numel, dtype, a_layout, b_layout);
}
void DivCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout, CUDALayout b_layout) {
    Launch<Op::Div>(a, b, output, numel, dtype, a_layout, b_layout);
}
void NegCUDA(const void* a, void* output, std::size_t numel, Dtype dtype,
             CUDALayout a_layout) {
    Launch<Op::Neg>(a, nullptr, output, numel, dtype, a_layout, CUDALayout{});
}

void SqrtCUDA(const void* input, void* output, Dtype dtype, std::size_t numel,
              CUDALayout layout) {
    if (numel == 0) return;
    switch (dtype) {
        case Dtype::kFloat32: LaunchSqrt<float, float>(input, output, numel, layout); break;
        case Dtype::kFloat16: LaunchSqrt<__half, __half>(input, output, numel, layout); break;
        case Dtype::kBFloat16: LaunchSqrt<__nv_bfloat16, __nv_bfloat16>(input, output, numel, layout); break;
        case Dtype::kInt8: LaunchSqrt<std::int8_t, float>(input, output, numel, layout); break;
        case Dtype::kUInt8: LaunchSqrt<std::uint8_t, float>(input, output, numel, layout); break;
        case Dtype::kInt16: LaunchSqrt<std::int16_t, float>(input, output, numel, layout); break;
        case Dtype::kInt32: LaunchSqrt<std::int32_t, float>(input, output, numel, layout); break;
        case Dtype::kInt64: LaunchSqrt<std::int64_t, float>(input, output, numel, layout); break;
        default: throw std::invalid_argument("sqrt: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

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

template <bool Log>
static void ExpLogCUDA(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchExpLog<float, float, Log>(x, out); break;
        case Dtype::kFloat16: LaunchExpLog<__half, __half, Log>(x, out); break;
        case Dtype::kBFloat16: LaunchExpLog<__nv_bfloat16, __nv_bfloat16, Log>(x, out); break;
        case Dtype::kInt8: LaunchExpLog<std::int8_t, float, Log>(x, out); break;
        case Dtype::kUInt8: LaunchExpLog<std::uint8_t, float, Log>(x, out); break;
        case Dtype::kInt16: LaunchExpLog<std::int16_t, float, Log>(x, out); break;
        case Dtype::kInt32: LaunchExpLog<std::int32_t, float, Log>(x, out); break;
        case Dtype::kInt64: LaunchExpLog<std::int64_t, float, Log>(x, out); break;
        default: throw std::invalid_argument(Log ? "log: unsupported dtype" : "exp: unsupported dtype");
    }
}

void ExpCUDA(const Tensor& x, Tensor& out) {
    ExpLogCUDA<false>(x, out);
}

void LogCUDA(const Tensor& x, Tensor& out) {
    ExpLogCUDA<true>(x, out);
}

namespace {

template <typename T>
__global__ void RsqrtBackwardKernel(const T* x, const T* grad, T* dx,
                                      std::size_t count, CUDALayout x_layout, CUDALayout grad_layout) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = GetNumelIndex(i, x_layout);
    const auto gi = GetNumelIndex(i, grad_layout);
    const float value = static_cast<float>(x[xi]);
    const float upstream = static_cast<float>(grad[gi]);
    const float y = rsqrtf(value);
    dx[i] = static_cast<T>(((-0.5f * y) * upstream) * y * y);
}

template <typename T>
void LaunchRsqrtBackward(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    LaunchBackward<T, RsqrtBackwardKernel<T>>(x, output_grad, dx);
}

}  // namespace

void RsqrtBackwardCUDA(const Tensor& x, const Tensor& output_grad, Tensor& dx) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchRsqrtBackward<float>(x, output_grad, dx); break;
        case Dtype::kFloat16: LaunchRsqrtBackward<__half>(x, output_grad, dx); break;
        case Dtype::kBFloat16: LaunchRsqrtBackward<__nv_bfloat16>(x, output_grad, dx); break;
        default: throw std::invalid_argument("rsqrt backward: unsupported dtype");
    }
}

}  // namespace mtensor
