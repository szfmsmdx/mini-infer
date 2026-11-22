#pragma once

#include <stdexcept>

namespace mtensor {

struct ModelConfig {
    int vocab_size = 0;
    int context_length = 0;
    int channels = 0;
    int heads = 0;
    int layers = 0;

protected:
    void check_positive_dimensions(const char* message) const {
        if (vocab_size <= 0 || context_length <= 0 || channels <= 0 || heads <= 0 || layers <= 0) {
            throw std::invalid_argument(message);
        }
    }
};

}  // namespace mtensor
