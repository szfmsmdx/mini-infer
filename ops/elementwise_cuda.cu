#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ops/elementwise.h"
#include "utils/memory.h"

namespace mtensor {
namespace {
enum class Op { Add, Sub, Mul, Div, Neg };

__device__ std::size_t GetNumelIndex(
    std::size_t idx, const int* shape, const std::size_t* stride, int ndim
){
    std::size_t offset = 0;
    for(int d = ndim - 1; d >= 0; d--){
        std::size_t cur_offset = idx % shape[d];
        idx /= shape[d];
        offset += cur_offset * stride[d];
    }
    return offset;
}

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
}

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
}
