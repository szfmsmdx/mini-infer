#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mini_infer/autograd.h"
#include "mini_infer/models/gpt_checkpoint.h"

inline mtensor::Device GptDevice(int argc, char** argv, int position) {
#ifdef ENABLE_CUDA
    const std::string_view name = argc > position ? argv[position] : "cuda";
#else
    const std::string_view name = argc > position ? argv[position] : "cpu";
#endif
    if (name == "cpu") return mtensor::Device::kCPU;
    if (name == "cuda") return mtensor::Device::kCUDA;
    throw std::invalid_argument("Device must be cpu or cuda");
}

inline std::string GenerateGPT(mtensor::GPT& model, const mtensor::Tokenizer& tokenizer,
                               const std::string& prompt, mtensor::Device device, std::mt19937& rng) {
    using namespace mtensor;
    NogradNode no_grad;
    auto tokens = tokenizer.encode(prompt);
    const std::size_t prompt_length = tokens.size();
    if (prompt_length == 0) throw std::invalid_argument("Enter a prompt");
    const auto context = static_cast<std::size_t>(model.config().context_length);
    const int vocab = tokenizer.vocab_size();
    std::vector<double> weights(vocab);

    for (int i = 0; i < 160; ++i) {
        const int length = static_cast<int>(std::min(context, tokens.size()));
        const Tensor input(tokens.data() + tokens.size() - length, {length}, Dtype::kInt32, device);
        const Tensor last = model(input)[length - 1].to(Device::kCPU);
        const auto* scores = static_cast<const float*>(last.data_ptr());
        const float max_score = *std::max_element(scores, scores + vocab);
        double total = 0.0;
        for (int token = 0; token < vocab; ++token) {
            weights[token] = std::exp(scores[token] - max_score);
            total += weights[token];
        }
        if (!std::isfinite(total) || total <= 0.0) {
            throw std::runtime_error("Invalid logits during generation");
        }
        std::discrete_distribution<std::int32_t> sample(weights.begin(), weights.end());
        tokens.push_back(sample(rng));
    }
    return tokenizer.decode({tokens.begin() + prompt_length, tokens.end()});
}
