#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include "mini_infer/tokenizer/bpe_tokenizer.h"

namespace mtensor {
namespace {

using TokenId = BPETokenizer::TokenId;
using Merge = BPETokenizer::Merge;
constexpr int kByteCount = 256;

bool IsSpace(unsigned char byte) {
    return byte == ' ' || (byte >= '\t' && byte <= '\r');
}

std::string_view NextPiece(std::string_view& text) {
    // 训练和编码按同样的规则切分。取出开头连续的空白或非空白字节，再把 text 移到下一段。
    const bool space = IsSpace(text.front());
    std::size_t end = 1;
    while (end < text.size() && IsSpace(text[end]) == space) ++end;
    const auto piece = text.substr(0, end);
    text.remove_prefix(end);
    return piece;
}

void AppendByteIds(std::vector<TokenId>& ids, std::string_view piece) {
    // 每个字节的无符号值（0 到 255）就是它的基础 token ID。
    for (unsigned char byte : piece) {
        ids.push_back(byte);
    }
}

void MergePair(std::vector<TokenId>& ids, Merge pair, TokenId replacement,
               std::size_t start = 0) {
    // 从 start 开始原地合并，命中时读两个、写一个，最后截掉尾部残留的旧 ID。
    std::size_t written = start;
    for (std::size_t read = start; read < ids.size(); ++read) {
        if (read + 1 < ids.size() && ids[read] == pair.first && ids[read + 1] == pair.second) {
            ids[written++] = replacement;
            ++read;
        } else {
            ids[written++] = ids[read];
        }
    }
    ids.resize(written);
}

struct Piece {
    std::vector<TokenId> ids;
    std::size_t frequency = 0;
};
}  // namespace

BPETokenizer::BPETokenizer(const std::vector<Merge>& merges) {
    for (int byte = 0; byte < kByteCount; ++byte) {
        // emplace_back(...args) -> 这里 args 就是 vector 存储类型构造函数所需的参数
        vocabulary_.emplace_back(1, static_cast<char>(byte));
    }
    for (const auto& pair : merges) AddMerge(pair);
}

BPETokenizer::BPETokenizer(std::string_view corpus, int target_vocab_size)
    : BPETokenizer(std::vector<Merge>{}) {
    if (target_vocab_size < kByteCount) {
        throw std::invalid_argument("BPE vocabulary must include all 256 bytes");
    }
    if (target_vocab_size == kByteCount) return;

    // 每种片段只转一次 ID，后续统计相邻对时按片段的出现次数加权。
    std::unordered_map<std::string_view, Piece> pieces;
    while (!corpus.empty()) {
        const auto text = NextPiece(corpus);
        if (text.size() < 2) continue;

        Piece& piece = pieces[text];
        if (piece.frequency == 0) {
            piece.ids.reserve(text.size());
            AppendByteIds(piece.ids, text);
        }
        ++piece.frequency;
    }

    while (vocab_size() < target_vocab_size) {
        // 1. 按合并后的切分重新统计相邻对。
        std::map<Merge, std::size_t> pair_frequencies;
        for (const auto& [text, piece] : pieces) {
            for (std::size_t i = 1; i < piece.ids.size(); ++i) {
                const Merge pair{piece.ids[i - 1], piece.ids[i]};
                pair_frequencies[pair] += piece.frequency;
            }
        }
        if (pair_frequencies.empty()) break;

        // 2. 选最高频对。同频时选字典序最小的 ID 对。
        Merge best_pair{};
        std::size_t best_frequency = 0;
        for (const auto& [pair, frequency] : pair_frequencies) {
            if (frequency > best_frequency) {
                best_pair = pair;
                best_frequency = frequency;
            }
        }

        // 3. 分配一个新 ID，在语料片段中执行合并。
        const TokenId replacement = vocab_size();
        AddMerge(best_pair);
        for (auto& [text, piece] : pieces) {
            MergePair(piece.ids, best_pair, replacement);
        }
    }
}

void BPETokenizer::AddMerge(Merge pair) {
    // 给相邻 token 对分配新 ID，记录合并后的字节串和规则顺序。
    const TokenId id = vocab_size();
    if (pair.first < 0 || pair.second < 0 || pair.first >= id || pair.second >= id) {
        throw std::invalid_argument("BPE merge refers to a token outside the preceding vocabulary");
    }
    auto insert_result = merge_ids_.emplace(pair, id);
    bool inserted = insert_result.second;
    if (!inserted) {
        throw std::invalid_argument("Duplicate BPE merge");
    }
    vocabulary_.push_back(vocabulary_[pair.first] + vocabulary_[pair.second]);
    merges_.push_back(pair);
}

int BPETokenizer::vocab_size() const {
    return static_cast<int>(vocabulary_.size());
}

std::vector<BPETokenizer::TokenId> BPETokenizer::encode(std::string_view text) const {
    std::vector<TokenId> ids;
    ids.reserve(text.size());
    while (!text.empty()) {
        // 当前片段追加到末尾，后面的合并只处理从 piece_start 开始的 ID。
        const std::size_t piece_start = ids.size();
        const auto piece = NextPiece(text);
        AppendByteIds(ids, piece);

        for (;;) {
            TokenId first_merge = vocab_size();
            for (std::size_t i = piece_start + 1; i < ids.size(); ++i) {
                const Merge pair{ids[i - 1], ids[i]};
                const auto it = merge_ids_.find(pair);
                if (it != merge_ids_.end()) first_merge = std::min(first_merge, it->second);
            }
            if (first_merge == vocab_size()) break;
            const Merge pair = merges_[first_merge - kByteCount];
            MergePair(ids, pair, first_merge, piece_start);
        }
    }
    return ids;
}

std::string BPETokenizer::decode(const std::vector<TokenId>& ids) const {
    std::string text;
    for (TokenId id : ids) {
        text.append(vocabulary_.at(id));
    }
    return text;
}

}  // namespace mtensor
