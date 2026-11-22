#include <cstddef>
#include <cuda_runtime.h>

#include "mini_infer/ops/attention.h"
#include "device/memory.h"

namespace mtensor {
namespace {

constexpr unsigned int kThreads = 128;

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>((count + kThreads - 1) / kThreads);
}

__device__ float Dot(const float* a, const float* b, int width) {
    float sum = 0.0f;
    for (int i = 0; i < width; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__device__ float Maximum(const float* row, int width) {
    float maximum = row[0];
    for (int i = 1; i < width; ++i) {
        maximum = fmaxf(maximum, row[i]);
    }
    return maximum;
}

// 每个线程计算一个头的一行，只对当前位置及之前的 token 归一化。
__global__ void CausalAttentionKernel(const float* q, const float* k, const float* v,
                                      int length, int width, int heads,
                                      float* probabilities, float* output) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int head = static_cast<int>(i / length);
    if (head >= heads) return;
    const int row = static_cast<int>(i % length);
    const int head_width = width / heads;
    const int head_start = head * head_width;
    const float scale = rsqrtf(static_cast<float>(head_width));
    const std::size_t offset = static_cast<std::size_t>(row) * width + head_start;
    float* p = probabilities + i * length;
    for (int j = 0; j <= row; ++j) {
        const float* k_row = k + static_cast<std::size_t>(j) * width + head_start;
        p[j] = Dot(q + offset, k_row, head_width) * scale;
    }
    const float maximum = Maximum(p, row + 1);
    float sum = 0.0f;
    for (int j = 0; j <= row; ++j) {
        p[j] = expf(p[j] - maximum);
        sum += p[j];
    }
    for (int col = 0; col < head_width; ++col) {
        output[offset + col] = 0.0f;
    }
    for (int j = 0; j <= row; ++j) {
        p[j] /= sum;
        const float* v_row = v + static_cast<std::size_t>(j) * width + head_start;
        for (int col = 0; col < head_width; ++col) {
            output[offset + col] += p[j] * v_row[col];
        }
    }
}

__global__ void CausalAttentionGradKernel(const float* q, const float* k, const float* v,
                                          int length, int width, int heads,
                                          const float* probabilities, const float* grad,
                                          float* dq, float* dk, float* dv) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int head = static_cast<int>(i / length);
    if (head >= heads) return;
    const int row = static_cast<int>(i % length);
    const int head_width = width / heads;
    const int head_start = head * head_width;
    const float scale = rsqrtf(static_cast<float>(head_width));
    const std::size_t offset = static_cast<std::size_t>(row) * width + head_start;
    const float* grad_row = grad + offset;
    const float* p = probabilities + i * length;
    float mean_dp = 0.0f;
    for (int j = 0; j <= row; ++j) {
        const float* v_row = v + static_cast<std::size_t>(j) * width + head_start;
        mean_dp += p[j] * Dot(grad_row, v_row, head_width);
    }
    for (int col = 0; col < head_width; ++col) {
        dq[offset + col] = 0.0f;
    }
    for (int j = 0; j <= row; ++j) {
        const std::size_t source = static_cast<std::size_t>(j) * width + head_start;
        const float dp = Dot(grad_row, v + source, head_width);
        const float ds = scale * p[j] * (dp - mean_dp);
        for (int col = 0; col < head_width; ++col) {
            dq[offset + col] += ds * k[source + col];
            atomicAdd(dk + source + col, ds * q[offset + col]);
            atomicAdd(dv + source + col, p[j] * grad_row[col]);
        }
    }
}

}  // namespace

void CausalAttentionCUDA(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                         Tensor& probabilities, Tensor& output) {
    const std::size_t rows = static_cast<std::size_t>(q.size(0)) * heads;
    const auto* q_data = static_cast<const float*>(q.data_ptr());
    const auto* k_data = static_cast<const float*>(k.data_ptr());
    const auto* v_data = static_cast<const float*>(v.data_ptr());
    auto* probs = static_cast<float*>(probabilities.data_ptr());
    auto* result = static_cast<float*>(output.data_ptr());
    CausalAttentionKernel<<<Blocks(rows), kThreads>>>(
        q_data, k_data, v_data, q.size(0), q.size(1), heads, probs, result);
    CUDA_CHECK(cudaGetLastError());
}

void CausalAttentionGradCUDA(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                             const Tensor& probabilities, const Tensor& grad,
                             Tensor& dq, Tensor& dk, Tensor& dv) {
    const std::size_t rows = static_cast<std::size_t>(q.size(0)) * heads;
    const auto* q_data = static_cast<const float*>(q.data_ptr());
    const auto* k_data = static_cast<const float*>(k.data_ptr());
    const auto* v_data = static_cast<const float*>(v.data_ptr());
    const auto* probs = static_cast<const float*>(probabilities.data_ptr());
    const auto* upstream = static_cast<const float*>(grad.data_ptr());
    auto* dq_data = static_cast<float*>(dq.data_ptr());
    auto* dk_data = static_cast<float*>(dk.data_ptr());
    auto* dv_data = static_cast<float*>(dv.data_ptr());
    CausalAttentionGradKernel<<<Blocks(rows), kThreads>>>(
        q_data, k_data, v_data, q.size(0), q.size(1), heads, probs, upstream,
        dq_data, dk_data, dv_data);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mtensor
