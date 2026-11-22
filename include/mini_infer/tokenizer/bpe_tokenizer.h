#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mtensor {

class BPETokenizer {
public:
    using TokenId = std::int32_t;
    using Merge = std::pair<TokenId, TokenId>;

    // 基础 ID 0–255 对应原始字节，新 ID 按合并顺序分配。
    BPETokenizer(std::string_view corpus, int target_vocab_size);
    explicit BPETokenizer(const std::vector<Merge>& merges);

    int vocab_size() const;
    const std::vector<Merge>& merges() const { return merges_; }
    std::vector<TokenId> encode(std::string_view text) const;
    std::string decode(const std::vector<TokenId>& ids) const;

private:
    void AddMerge(Merge pair);

    std::vector<std::string> vocabulary_;
    std::vector<Merge> merges_;
    std::map<Merge, TokenId> merge_ids_;
};

}  // namespace mtensor
