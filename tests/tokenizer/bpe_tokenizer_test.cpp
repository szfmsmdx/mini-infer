#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_infer/tokenizer/bpe_tokenizer.h"

using mtensor::BPETokenizer;
using Ids = std::vector<BPETokenizer::TokenId>;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        const BPETokenizer bpe("cabca cab ca", 258);
        const std::vector<BPETokenizer::Merge> expected{{'c', 'a'}, {256, 'b'}};
        Check(bpe.merges() == expected, "BPE must count pairs after each merge");
        Check(bpe.encode("caba") == Ids({257, 'a'}), "BPE must apply ranked merges");
        Check(bpe.decode({257, 'a'}) == "caba", "BPE must concatenate decoded pieces");

        const BPETokenizer restored(bpe.merges());
        const std::string bytes = std::string("cab \n") + '\0' + static_cast<char>(0xff);
        Check(restored.encode(bytes) == bpe.encode(bytes), "Restored BPE must preserve token IDs");
        Check(restored.decode(restored.encode(bytes)) == bytes, "BPE must round trip every byte");
        Check(restored.encode("").empty() && restored.decode({}).empty(), "Empty input round trip");

        const BPETokenizer repeats("aaaa", 257);
        Check(repeats.encode("aaaa") == Ids({256, 256}), "Overlapping pairs merge left to right");
        Check(repeats.encode("a a") == Ids({'a', ' ', 'a'}), "BPE must respect piece boundaries");

        const BPETokenizer tied("bc ac", 257);
        Check(tied.merges().front() == BPETokenizer::Merge{'a', 'c'}, "BPE tie order must be deterministic");
        const BPETokenizer ranked(std::vector<BPETokenizer::Merge>{{'b', 'c'}, {'a', 'b'}});
        Check(ranked.encode("abc") == Ids({'a', 256}), "Earlier BPE rule must win at overlapping positions");

        const BPETokenizer boundaries(std::vector<BPETokenizer::Merge>{{'a', ' '}, {'a', 'a'}});
        Check(boundaries.encode("aa aa a") == Ids({257, ' ', 257, ' ', 'a'}),
              "Encoding a new piece must preserve earlier IDs and respect piece boundaries");

        const BPETokenizer short_corpus("ab", 10000);
        Check(short_corpus.vocab_size() == 257, "BPE must stop once no pair remains");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
