#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_infer/layers/embedding.h"
#include "mini_infer/ops/embedding.h"
#include "init/weight_init.h"

namespace mtensor {

Embedding::Embedding(int vocab_size, int channels, Device device, WeightInitializer initializer)
    : vocab_size_(vocab_size), channels_(channels), device_(device),
      initializer_(std::move(initializer)) {
    init();
}

void Embedding::init() {
    if (vocab_size_ <= 0 || channels_ <= 0) {
        throw std::invalid_argument("Embedding: vocab_size and channels must be positive");
    }

    std::vector<float> values(static_cast<std::size_t>(vocab_size_) * channels_);
    if (initializer_) {
        initializer_(values.data(), values.size());
        initializer_ = {};
    } else {
        thread_local std::mt19937 rng(std::random_device{}());
        weight_init::normal_(values.data(), values.size(), rng);
    }
    weight_ = Tensor(values.data(), {vocab_size_, channels_}, Dtype::kFloat32, device_);
    weight_.requires_grad_(true);
    register_parameter(weight_);
}

Tensor Embedding::forward(const Tensor& indices) {
    return embedding(weight_, indices);
}

}  // namespace mtensor
