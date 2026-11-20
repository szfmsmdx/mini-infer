#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ops/backend.h"
#include "mini_infer/tensor.h"
#include "ops/cuda/index.cuh"
#include "utils/memory.h"

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
                if constexpr (op == Op::Mul) bits = static_cast<U>(static_cast<unsigned long long>(ua) * ub);
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
                                  const int* shape, const std::size_t* a_stride,
                                  const std::size_t* b_stride, int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    const T av = a[GetNumelIndex(idx, shape, a_stride, ndim)];
    T bv{};
    if constexpr (op != Op::Neg) bv = b[GetNumelIndex(idx, shape, b_stride, ndim)];
    out[idx] = Calculate<T, op>(av, bv);
}

template <typename Half, Op op>
__global__ void HalfKernel(const std::uint16_t* a, const std::uint16_t* b,
                           std::uint16_t* out, int numel, const int* shape,
                           const std::size_t* a_stride, const std::size_t* b_stride, int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    const std::uint16_t abits = a[GetNumelIndex(idx, shape, a_stride, ndim)];
    if constexpr (op == Op::Neg) {
        out[idx] = abits ^ 0x8000u;
    } else {
        const std::uint16_t bbits = b[GetNumelIndex(idx, shape, b_stride, ndim)];
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
                 const int* shape, const std::size_t* a_stride, const std::size_t* b_stride,
                 int ndim, unsigned blocks, unsigned threads) {
    ElementwiseKernel<T, op><<<blocks, threads>>>(static_cast<const T*>(a),
        static_cast<const T*>(b), static_cast<T*>(output), numel, shape, a_stride, b_stride, ndim);
}

template <Op op>
void Launch(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
            const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim) {
    if (numel == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((numel - 1) / threads + 1);
    switch (dtype) {
        case Dtype::kFloat32: LaunchTyped<float, op>(a,b,output,numel,shape,a_stride,b_stride,ndim,blocks,threads); break;
        case Dtype::kInt8: LaunchTyped<std::int8_t, op>(a,b,output,numel,shape,a_stride,b_stride,ndim,blocks,threads); break;
        case Dtype::kUInt8: LaunchTyped<std::uint8_t, op>(a,b,output,numel,shape,a_stride,b_stride,ndim,blocks,threads); break;
        case Dtype::kInt16: LaunchTyped<std::int16_t, op>(a,b,output,numel,shape,a_stride,b_stride,ndim,blocks,threads); break;
        case Dtype::kInt32: LaunchTyped<std::int32_t, op>(a,b,output,numel,shape,a_stride,b_stride,ndim,blocks,threads); break;
        case Dtype::kInt64: LaunchTyped<std::int64_t, op>(a,b,output,numel,shape,a_stride,b_stride,ndim,blocks,threads); break;
        case Dtype::kFloat16: HalfKernel<__half, op><<<blocks, threads>>>(static_cast<const std::uint16_t*>(a), static_cast<const std::uint16_t*>(b), static_cast<std::uint16_t*>(output), numel, shape, a_stride, b_stride, ndim); break;
        case Dtype::kBFloat16: HalfKernel<__nv_bfloat16, op><<<blocks, threads>>>(static_cast<const std::uint16_t*>(a), static_cast<const std::uint16_t*>(b), static_cast<std::uint16_t*>(output), numel, shape, a_stride, b_stride, ndim); break;
        default: throw std::invalid_argument("unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Input, typename Output>
__global__ void SqrtKernel(const Input* input, Output* output, std::size_t numel,
                           const int* shape, const std::size_t* stride, int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float x = static_cast<float>(input[GetNumelIndex(i, shape, stride, ndim)]);
    output[i] = static_cast<Output>(sqrtf(x));
}

template <typename Input, typename Output>
void LaunchSqrt(const void* input, void* output, std::size_t numel,
                const int* shape, const std::size_t* stride, int ndim) {
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((numel - 1) / threads + 1);
    SqrtKernel<<<blocks, threads>>>(static_cast<const Input*>(input), static_cast<Output*>(output),
                                    numel, shape, stride, ndim);
}

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

// 连续输入直接用 i；视图按各自的 stride 定位。输出总是连续。
template <typename In, typename Out>
__global__ void ExpKernel(const In* x, Out* out, std::size_t count, const int* shape,
                            const std::size_t* xs, int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = xs ? GetNumelIndex(i, shape, xs, ndim) : i;
    const float y = expf(static_cast<float>(x[xi]));
    out[i] = static_cast<Out>(y);
}

template <typename In, typename Out>
void LaunchExp(const Tensor& x, Tensor& out) {
    const auto n = x.numel();
    if (n == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n - 1) / threads + 1);
    ExpKernel<In, Out><<<blocks, threads>>>(
        static_cast<const In*>(x.data_ptr()), static_cast<Out*>(out.data_ptr()), n, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}


template <typename In, typename Out>
__global__ void SigmoidKernel(const In* x, Out* out, std::size_t count, const int* shape,
                            const std::size_t* xs, int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = xs ? GetNumelIndex(i, shape, xs, ndim) : i;
    const float v = static_cast<float>(x[xi]);
    const float t = expf(-fabsf(v));
    out[i] = static_cast<Out>(v >= 0.0f ? 1.0f / (1.0f + t) : t / (1.0f + t));
}

template <typename In, typename Out>
void LaunchSigmoid(const Tensor& x, Tensor& out) {
    const auto n = x.numel();
    if (n == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n - 1) / threads + 1);
    SigmoidKernel<In, Out><<<blocks, threads>>>(
        static_cast<const In*>(x.data_ptr()), static_cast<Out*>(out.data_ptr()), n, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}


template <typename In, typename Out>
__global__ void SiluKernel(const In* x, Out* out, std::size_t count, const int* shape,
                           const std::size_t* xs, int ndim) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xi = xs ? GetNumelIndex(i, shape, xs, ndim) : i;
    const float v = static_cast<float>(x[xi]);
    float y;
    if (v < -80.0f) {
        // 负尾部用 double，避免 exp(x) 在乘 x 前下溢。
        const double t = exp(static_cast<double>(v));
        y = static_cast<float>(v * t / (1.0 + t));
    } else {
        const float t = expf(-fabsf(v));
        y = v >= 0.0f ? v / (1.0f + t) : (v * t) / (1.0f + t);
    }
    out[i] = static_cast<Out>(y);
}

template <typename In, typename Out>
void LaunchSilu(const Tensor& x, Tensor& out) {
    const auto n = x.numel();
    if (n == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n - 1) / threads + 1);
    SiluKernel<In, Out><<<blocks, threads>>>(
        static_cast<const In*>(x.data_ptr()), static_cast<Out*>(out.data_ptr()), n, x.shape_cuda(),
        x.is_contiguous() ? nullptr : x.stride_cuda(), x.dim());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void AddCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim) {
    Launch<Op::Add>(a, b, output, numel, dtype, shape, a_stride, b_stride, ndim);
}
void SubCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim) {
    Launch<Op::Sub>(a, b, output, numel, dtype, shape, a_stride, b_stride, ndim);
}
void MulCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim) {
    Launch<Op::Mul>(a, b, output, numel, dtype, shape, a_stride, b_stride, ndim);
}
void DivCUDA(const void* a, const void* b, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, const std::size_t* b_stride, int ndim) {
    Launch<Op::Div>(a, b, output, numel, dtype, shape, a_stride, b_stride, ndim);
}
void NegCUDA(const void* a, void* output, std::size_t numel, Dtype dtype,
             const int* shape, const std::size_t* a_stride, int ndim) {
    Launch<Op::Neg>(a, nullptr, output, numel, dtype, shape, a_stride, nullptr, ndim);
}

void SqrtCUDA(const void* input, void* output, Dtype dtype, std::size_t numel,
              const int* shape, const std::size_t* stride, int ndim) {
    if (numel == 0) return;
    switch (dtype) {
        case Dtype::kFloat32: LaunchSqrt<float, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kFloat16: LaunchSqrt<__half, __half>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kBFloat16: LaunchSqrt<__nv_bfloat16, __nv_bfloat16>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kInt8: LaunchSqrt<std::int8_t, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kUInt8: LaunchSqrt<std::uint8_t, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kInt16: LaunchSqrt<std::int16_t, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kInt32: LaunchSqrt<std::int32_t, float>(input, output, numel, shape, stride, ndim); break;
        case Dtype::kInt64: LaunchSqrt<std::int64_t, float>(input, output, numel, shape, stride, ndim); break;
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

void ExpCUDA(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchExp<float, float>(x, out); break;
        case Dtype::kFloat16: LaunchExp<__half, __half>(x, out); break;
        case Dtype::kBFloat16: LaunchExp<__nv_bfloat16, __nv_bfloat16>(x, out); break;
        case Dtype::kInt8: LaunchExp<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: LaunchExp<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: LaunchExp<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: LaunchExp<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: LaunchExp<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("exp: unsupported dtype");
    }
}

void SigmoidCUDA(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchSigmoid<float, float>(x, out); break;
        case Dtype::kFloat16: LaunchSigmoid<__half, __half>(x, out); break;
        case Dtype::kBFloat16: LaunchSigmoid<__nv_bfloat16, __nv_bfloat16>(x, out); break;
        case Dtype::kInt8: LaunchSigmoid<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: LaunchSigmoid<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: LaunchSigmoid<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: LaunchSigmoid<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: LaunchSigmoid<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("sigmoid: unsupported dtype");
    }
}


void SiluCUDA(const Tensor& x, Tensor& out) {
    switch (x.dtype()) {
        case Dtype::kFloat32: LaunchSilu<float, float>(x, out); break;
        case Dtype::kFloat16: LaunchSilu<__half, __half>(x, out); break;
        case Dtype::kBFloat16: LaunchSilu<__nv_bfloat16, __nv_bfloat16>(x, out); break;
        case Dtype::kInt8: LaunchSilu<std::int8_t, float>(x, out); break;
        case Dtype::kUInt8: LaunchSilu<std::uint8_t, float>(x, out); break;
        case Dtype::kInt16: LaunchSilu<std::int16_t, float>(x, out); break;
        case Dtype::kInt32: LaunchSilu<std::int32_t, float>(x, out); break;
        case Dtype::kInt64: LaunchSilu<std::int64_t, float>(x, out); break;
        default: throw std::invalid_argument("silu: unsupported dtype");
    }
}

}  // namespace mtensor
