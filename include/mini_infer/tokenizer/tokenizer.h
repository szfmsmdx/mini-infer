#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mtensor {

// 按语料中出现的字节建词表，token ID 按无符号字节值排序。
class Tokenizer {
public:
    explicit Tokenizer(const std::string& corpus);

    int vocab_size() const;
    const std::string& vocabulary() const;
    std::vector<std::int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::int32_t>& ids) const;

private:
    std::array<std::int32_t, 256> byte_to_id_;
    std::string vocabulary_;
};

}  // namespace mtensor
