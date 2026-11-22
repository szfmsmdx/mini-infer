#pragma once

#include <filesystem>

#include "mini_infer/models/gpt.h"
#include "mini_infer/tokenizer/tokenizer.h"

namespace mtensor {

struct GPTCheckpoint {
    GPT model;
    Tokenizer tokenizer;
};

void SaveGPTCheckpoint(const std::filesystem::path& path, const GPT& model,
                       const Tokenizer& tokenizer);
GPTCheckpoint LoadGPTCheckpoint(const std::filesystem::path& path, Device device);

}  // namespace mtensor
