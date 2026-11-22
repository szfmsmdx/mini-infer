#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_infer/tokenizer/tokenizer.h"

int main() {
    try {
        const char bytes[] = {'b', 'A', static_cast<char>(0xff), '\0', 'A'};
        const std::string corpus(bytes, sizeof(bytes));
        const mtensor::Tokenizer tokenizer(corpus);
        const std::vector<std::int32_t> expected{2, 1, 3, 0, 1};
        if (tokenizer.vocab_size() != 4 || tokenizer.encode(corpus) != expected ||
            tokenizer.decode(expected) != corpus) {
            throw std::runtime_error("Tokenizer byte ordering or round trip");
        }
        const mtensor::Tokenizer restored(tokenizer.vocabulary());
        if (restored.encode(corpus) != expected || restored.decode(expected) != corpus) {
            throw std::runtime_error("Tokenizer restored from its vocabulary");
        }

        bool rejected = false;
        try {
            tokenizer.encode("unknown");
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Tokenizer accepted a byte outside the vocabulary");
        }

        rejected = false;
        try {
            tokenizer.decode({-1});
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Tokenizer accepted a token outside the vocabulary");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
