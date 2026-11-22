#pragma once

#include <limits>
#include <stdexcept>

namespace mtensor {

inline int HeadWidth(int channels, int heads,
                     const char* message = "channels and heads must be positive, and channels must be divisible by heads") {
    if (channels <= 0 || heads <= 0 || channels % heads != 0) {
        throw std::invalid_argument(message);
    }
    return channels / heads;
}

// 当前 decoder 使用成对旋转的 RoPE 和 4 倍宽度的 MLP。
inline int CheckedDecoderHeads(int channels, int heads, const char* message) {
    if (channels > std::numeric_limits<int>::max() / 4 ||
        HeadWidth(channels, heads, message) % 2 != 0) {
        throw std::invalid_argument(message);
    }
    return heads;
}

}  // namespace mtensor
