#include <cstddef>
#include <stdexcept>

#include "mini_infer/tokenizer/tokenizer.h"

namespace mtensor {

Tokenizer::Tokenizer(const std::string& corpus) {
    if (corpus.empty()) {
        throw std::invalid_argument("Tokenizer requires a nonempty corpus");
    }

    std::array<bool, 256> present{};
    for (unsigned char byte : corpus) {
        present[byte] = true;
    }

    byte_to_id_.fill(-1);
    for (std::size_t byte = 0; byte < present.size(); ++byte) {
        if (present[byte]) {
            byte_to_id_[byte] = static_cast<std::int32_t>(vocabulary_.size());
            vocabulary_.push_back(static_cast<char>(byte));
        }
    }
}

int Tokenizer::vocab_size() const {
    return static_cast<int>(vocabulary_.size());
}

const std::string& Tokenizer::vocabulary() const {
    return vocabulary_;
}

std::vector<std::int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<std::int32_t> ids(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto byte = static_cast<unsigned char>(text[i]);
        const auto id = byte_to_id_[byte];
        if (id < 0) {
            throw std::invalid_argument("Byte outside tokenizer vocabulary: " + std::to_string(byte));
        }
        ids[i] = id;
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
    std::string text(ids.size(), '\0');
    for (std::size_t i = 0; i < ids.size(); ++i) {
        text[i] = vocabulary_.at(ids[i]);
    }
    return text;
}

}  // namespace mtensor
