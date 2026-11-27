#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#ifdef GPT_CHECKPOINT_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include "mini_infer/autograd.h"
#include "mini_infer/models/gpt_checkpoint.h"

namespace {

using namespace mtensor;
namespace fs = std::filesystem;

#ifdef GPT_CHECKPOINT_TEST_CUDA
constexpr Device kLoadDevice = Device::kCUDA;
#else
constexpr Device kLoadDevice = Device::kCPU;
#endif

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckTensor(const Tensor& actual, const Tensor& expected, const char* message) {
    Check(actual.shape() == expected.shape() && actual.dtype() == Dtype::kFloat32 &&
              actual.device() == Device::kCPU,
          message);
    Check(std::memcmp(actual.data_ptr(), expected.data_ptr(), expected.nbytes()) == 0, message);
}

void CheckV1Parameters(const GPT& model, std::uint32_t seed) {
    const auto& config = model.config();
    const auto parameters = static_cast<const Module&>(model).parameters();
    Check(parameters.size() == 8u * config.layers + 2u, "GPT v1 parameter count");

    // 按旧格式逐个核对。Q/K/V 形状相同，必须用不同的值验证顺序。
    std::mt19937 rng(seed);
    std::size_t next = 0;
    const auto check = [&](const std::vector<int>& shape, float stddev, const char* message) {
        std::size_t count = 1;
        for (int dimension : shape) count *= dimension;
        std::vector<float> expected(count, 1.0f);
        if (stddev > 0.0f) {
            std::normal_distribution<float> normal(0.0f, stddev);
            for (float& value : expected) value = normal(rng);
        }
        const Tensor& actual = parameters[next++];
        Check(actual.requires_grad(), "GPT parameter must retain gradients");
        CheckTensor(actual, Tensor(expected.data(), shape, Dtype::kFloat32), message);
    };

    constexpr float regular = 0.02f;
    const float residual = regular / std::sqrt(2.0f * config.layers);
    const int c = config.channels;
    check({config.vocab_size, c}, regular, "GPT v1 token embedding");
    for (int i = 0; i < config.layers; ++i) {
        check({c}, 0.0f, "GPT v1 attention norm");
        check({c, c}, regular, "GPT v1 Q weights");
        check({c, c}, regular, "GPT v1 K weights");
        check({c, c}, regular, "GPT v1 V weights");
        check({c, c}, residual, "GPT v1 attention projection");
        check({c}, 0.0f, "GPT v1 MLP norm");
        check({4 * c, c}, regular, "GPT v1 MLP expand");
        check({c, 4 * c}, residual, "GPT v1 MLP contract");
    }
    check({c}, 0.0f, "GPT v1 final norm");
}

void CheckModuleTraversal(const GPT& model) {
    const auto& children = model.named_children();
    Check(children.size() == static_cast<std::size_t>(model.config().layers) + 2,
          "GPT must enumerate only its direct children");
    Check(children.front().first == "token_embedding" && children[1].first == "decoder_0" &&
              children.back().first == "final_norm",
          "GPT child registration order");

    const auto& decoder_children = children[1].second->named_children();
    Check(decoder_children.size() == 8 && decoder_children[1].first == "q",
          "Decoder child registration order");
    const Module& q = *decoder_children[1].second;
    Check(q.named_children().empty(), "Linear weight is a parameter, not a child module");
    const auto q_params = q.parameters();
    Check(q_params.size() == 1, "Linear without bias has only one parameter");

    const auto parameters = model.parameters();
    Check(parameters.size() == 8u * model.config().layers + 2u, "GPT recursive parameter count");
    Check(parameters[2].data_ptr() == q_params.front().data_ptr(),
          "Recursive traversal must return the original child weight");

    std::ostringstream linear_out;
    q.print(linear_out);
    Check(linear_out.str() == "Linear\n", "Standalone module print must use its own name");

    std::ostringstream out;
    model.print(out);
    const auto description = out.str();
    Check(description.find("GPT\n  token_embedding: Embedding\n") == 0 &&
              description.find("\n  decoder_0: SimpleDecoderLayer\n") != std::string::npos &&
              description.find("\n    q: Linear\n") != std::string::npos &&
              description.find("weight") == std::string::npos,
          "Model print must show the module tree without its weights");
}

void TestCheckpoint(const fs::path& path) {
    const GPTConfig config{4, 6, 8, 2, 2};
    constexpr std::uint32_t seed = 7;
    GPT model(config, Device::kCPU, seed);
    NogradNode no_grad;
    CheckV1Parameters(model, seed);
    SaveGPTCheckpoint(path, model, Tokenizer("abcd"));

    Tensor final_scale = model.parameters().back();
    final_scale.fill_(0.5f);
    const Tokenizer tokenizer("dca\n");
    SaveGPTCheckpoint(path, model, tokenizer);
    for (const auto& entry : fs::directory_iterator(path.parent_path())) {
        Check(entry.path() == path, "Checkpoint save left a temporary file");
    }

    GPTCheckpoint restored = LoadGPTCheckpoint(path, kLoadDevice);
    CheckModuleTraversal(restored.model);
    const GPTConfig& loaded = restored.model.config();
    Check(loaded.vocab_size == config.vocab_size &&
              loaded.context_length == config.context_length &&
              loaded.channels == config.channels && loaded.heads == config.heads &&
              loaded.layers == config.layers,
          "Checkpoint model configuration");
    Check(restored.tokenizer.vocabulary() == tokenizer.vocabulary(), "Checkpoint vocabulary");

    const auto weights = model.parameters();
    const auto loaded_weights = restored.model.parameters();
    Check(loaded_weights.size() == weights.size(), "Checkpoint parameter count");
    for (std::size_t i = 0; i < weights.size(); ++i) {
        Check(loaded_weights[i].device() == kLoadDevice, "Checkpoint parameter is on the wrong device");
        CheckTensor(loaded_weights[i].to(Device::kCPU), weights[i], "Checkpoint parameter data or metadata");
    }

    const std::string text = "\nca";
    const auto ids = tokenizer.encode(text);
    Check(restored.tokenizer.encode(text) == ids && restored.tokenizer.decode(ids) == text,
          "Checkpoint token IDs");
    const Tensor prompt(ids.data(), {static_cast<int>(ids.size())}, Dtype::kInt32);
    const Tensor logits = restored.model(prompt.to(kLoadDevice));
    const Tensor reference = model(prompt);
    Check(logits.device() == kLoadDevice, "Checkpoint logits are on the wrong device");
    if (kLoadDevice == Device::kCPU) {
        CheckTensor(logits, reference, "Checkpoint logits");
    } else {
        const Tensor host = logits.to(Device::kCPU);
        Check(host.shape() == reference.shape() && host.dtype() == Dtype::kFloat32, "CUDA logits metadata");
        const auto* actual = static_cast<const float*>(host.data_ptr());
        const auto* expected = static_cast<const float*>(reference.data_ptr());
        for (std::size_t i = 0; i < reference.numel(); ++i) {
            Check(std::abs(actual[i] - expected[i]) <= 1e-5f + 1e-4f * std::abs(expected[i]),
                  "CUDA checkpoint logits differ from CPU");
        }
    }

    fs::resize_file(path, fs::file_size(path) - 1);
    bool rejected = false;
    try {
        (void)LoadGPTCheckpoint(path, kLoadDevice);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    Check(rejected, "Loader accepted a truncated checkpoint");
}

}  // namespace

int main() {
    namespace fs = std::filesystem;

    fs::path directory;
    bool owns_directory = false;
    int result = 0;
    try {
#ifdef GPT_CHECKPOINT_TEST_CUDA
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) return 77;
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const char* prefix = kLoadDevice == Device::kCUDA ? "mini-infer-gpt-cuda-" : "mini-infer-gpt-cpu-";
        directory = fs::temp_directory_path() / (prefix + std::to_string(stamp));
        owns_directory = fs::create_directory(directory);
        Check(owns_directory, "Checkpoint test directory already exists");
        TestCheckpoint(directory / "model.bin");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }

    if (owns_directory) {
        std::error_code error;
        fs::remove(directory / "model.bin", error);
        fs::remove(directory, error);
    }
    return result;
}
