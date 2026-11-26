#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "mini_infer/tensor/tensor.h"

namespace mtensor {
namespace {

float ReadMatrix(const Tensor& x, int row, int col) {
    const auto* data = static_cast<const float*>(x.data_ptr());
    return data[static_cast<std::size_t>(row) * x.stride()[0] +
                static_cast<std::size_t>(col) * x.stride()[1]];
}

}  // namespace

void CausalAttentionCPU(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                        Tensor& prob, Tensor& output) {
    const int length = q.size(0);
    const int channels = q.size(1);
    const int head_width = channels / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_width));
    auto* prob_data = static_cast<float*>(prob.data_ptr());
    auto* result = static_cast<float*>(output.data_ptr());
    std::fill(prob_data, prob_data + prob.numel(), 0.0f);
    std::vector<float> scores(length);
    for (int head = 0; head < heads; ++head) {
        const int begin = head * head_width;
        for (int row = 0; row < length; ++row) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (int key = 0; key <= row; ++key) {
                float dot = 0.0f;
                for (int col = 0; col < head_width; ++col) {
                    dot += ReadMatrix(q, row, begin + col) *
                           ReadMatrix(k, key, begin + col);
                }
                scores[key] = dot * scale;
                maximum = std::max(maximum, scores[key]);
            }
            float denominator = 0.0f;
            for (int key = 0; key <= row; ++key) {
                scores[key] = std::exp(scores[key] - maximum);
                denominator += scores[key];
            }
            for (int key = 0; key <= row; ++key) {
                const float p = scores[key] / denominator;
                prob_data[(static_cast<std::size_t>(head) * length + row) * length + key] = p;
            }
            for (int col = 0; col < head_width; ++col) {
                float value = 0.0f;
                for (int key = 0; key <= row; ++key) {
                    value += prob_data[(static_cast<std::size_t>(head) * length + row) *
                                       length + key] * ReadMatrix(v, key, begin + col);
                }
                result[static_cast<std::size_t>(row) * channels + begin + col] = value;
            }
        }
    }
}

void CausalAttentionGradCPU(const Tensor& q, const Tensor& k, const Tensor& v, int heads,
                            const Tensor& prob, const Tensor& grad,
                            Tensor& dq, Tensor& dk, Tensor& dv) {
    const int length = q.size(0);
    const int channels = q.size(1);
    const int head_width = channels / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_width));
    const auto* prob_data = static_cast<const float*>(prob.data_ptr());
    auto* q_grad = static_cast<float*>(dq.data_ptr());
    auto* k_grad = static_cast<float*>(dk.data_ptr());
    auto* v_grad = static_cast<float*>(dv.data_ptr());
    std::fill(q_grad, q_grad + dq.numel(), 0.0f);
    std::fill(k_grad, k_grad + dk.numel(), 0.0f);
    std::fill(v_grad, v_grad + dv.numel(), 0.0f);
    std::vector<float> dp(length);
    for (int head = 0; head < heads; ++head) {
        const int begin = head * head_width;
        for (int row = 0; row < length; ++row) {
            float weighted = 0.0f;
            for (int key = 0; key <= row; ++key) {
                float value = 0.0f;
                for (int col = 0; col < head_width; ++col) {
                    value += ReadMatrix(grad, row, begin + col) *
                             ReadMatrix(v, key, begin + col);
                    v_grad[static_cast<std::size_t>(key) * channels + begin + col] +=
                        prob_data[(static_cast<std::size_t>(head) * length + row) *
                                  length + key] * ReadMatrix(grad, row, begin + col);
                }
                dp[key] = value;
                weighted += prob_data[(static_cast<std::size_t>(head) * length + row) *
                                      length + key] * value;
            }
            for (int key = 0; key <= row; ++key) {
                const float p = prob_data[(static_cast<std::size_t>(head) * length + row) *
                                          length + key];
                const float ds = (p * scale) * (dp[key] - weighted);
                for (int col = 0; col < head_width; ++col) {
                    q_grad[static_cast<std::size_t>(row) * channels + begin + col] +=
                        ds * ReadMatrix(k, key, begin + col);
                    k_grad[static_cast<std::size_t>(key) * channels + begin + col] +=
                        ds * ReadMatrix(q, row, begin + col);
                }
            }
        }
    }
}

}  // namespace mtensor
