#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_infer/models/gpt_checkpoint.h"
#include "mini_infer/autograd.h"

namespace mtensor {
namespace {

constexpr std::array<char, 8> kMagic{'M', 'I', 'G', 'P', 'T', '0', '1', '\n'};
using Dimensions = std::array<std::int32_t, 5>;
constexpr std::size_t kHeaderSize = kMagic.size() + sizeof(std::int32_t) * 5;

void CheckCheckpointSize(const GPTConfig& config, std::uintmax_t bytes) {
    if (config.vocab_size < 2 || config.vocab_size > 256 ||
        config.context_length <= 0 || config.channels <= 0 ||
        config.channels > std::numeric_limits<int>::max() / 4 ||
        config.heads <= 0 || config.layers <= 0) {
        throw std::runtime_error("Invalid GPT checkpoint dimensions");
    }
    const auto prefix = kHeaderSize + static_cast<std::size_t>(config.vocab_size);
    const auto channels = static_cast<std::uintmax_t>(config.channels);
    const auto layers = static_cast<std::uintmax_t>(config.layers);
    const auto shared = channels * (config.vocab_size + 1);
    const auto per_layer = 12 * channels * channels + 2 * channels;

    // 比较文件中实际存放的元素数，乘以层数前先避免溢出。
    if (bytes < prefix || (bytes - prefix) % sizeof(float) != 0) {
        throw std::runtime_error("Invalid GPT checkpoint size");
    }
    const auto count = (bytes - prefix) / sizeof(float);
    if (count < shared || (count - shared) % layers != 0 ||
        (count - shared) / layers != per_layer) {
        throw std::runtime_error("GPT checkpoint size does not match its dimensions");
    }
}

}  // namespace

void SaveGPTCheckpoint(const std::filesystem::path& path, const GPT& model,
                       const Tokenizer& tokenizer) {
    const GPTConfig& config = model.config();
    if (tokenizer.vocab_size() != config.vocab_size) {
        throw std::invalid_argument("Tokenizer vocabulary does not match the model");
    }

    // 完整写入同目录临时文件后替换，磁盘上只保留最后一份完整权重。
    auto temporary_dir = path;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    temporary_dir += ".tmp-" + std::to_string(stamp) + "-" + std::to_string(std::random_device{}());
    if (!std::filesystem::create_directory(temporary_dir)) {
        throw std::runtime_error("Could not create temporary checkpoint directory");
    }
    const auto temporary = temporary_dir / "checkpoint";
    try {
        std::ofstream output;
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.open(temporary, std::ios::binary | std::ios::trunc);
        const Dimensions dims{config.vocab_size, config.context_length, config.channels,
                              config.heads, config.layers};
        output.write(kMagic.data(), kMagic.size());
        output.write(reinterpret_cast<const char*>(dims.data()), kHeaderSize - kMagic.size());
        const auto& vocabulary = tokenizer.vocabulary();
        output.write(vocabulary.data(), vocabulary.size());
        {
            NogradNode no_grad;
            for (const Tensor& parameter : model.parameters()) {
                const Tensor host = parameter.to(Device::kCPU);
                output.write(static_cast<const char*>(host.data_ptr()), host.nbytes());
            }
        }
        output.close();
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code error;
        std::filesystem::remove(temporary, error);
        std::filesystem::remove(temporary_dir, error);
        throw;
    }
    std::filesystem::remove(temporary_dir);
}

GPTCheckpoint LoadGPTCheckpoint(const std::filesystem::path& path, Device device) {
    std::ifstream input;
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.open(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    input.seekg(0);
    std::array<char, kMagic.size()> magic{};
    input.read(magic.data(), magic.size());
    if (magic != kMagic) throw std::runtime_error("Unrecognized GPT checkpoint");

    Dimensions dims{};
    input.read(reinterpret_cast<char*>(dims.data()), kHeaderSize - kMagic.size());
    const GPTConfig config{dims[0], dims[1], dims[2], dims[3], dims[4]};
    CheckCheckpointSize(config, static_cast<std::uintmax_t>(bytes));

    std::string vocabulary(config.vocab_size, '\0');
    input.read(vocabulary.data(), vocabulary.size());
    Tokenizer tokenizer(vocabulary);
    if (tokenizer.vocabulary() != vocabulary) {
        throw std::runtime_error("Invalid GPT checkpoint token order");
    }

    NogradNode no_grad;
    GPT model(config, device);
    std::vector<float> weights;
    for (Tensor parameter : model.parameters()) {
        weights.resize(parameter.numel());
        input.read(reinterpret_cast<char*>(weights.data()), parameter.nbytes());
        parameter.copy_(Tensor(weights.data(), parameter.shape(), Dtype::kFloat32, device));
    }
    return {std::move(model), std::move(tokenizer)};
}

}  // namespace mtensor
