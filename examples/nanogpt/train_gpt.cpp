#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gpt_generate.h"
#include "mini_infer/autograd.h"
#include "mini_infer/models/gpt.h"
#include "mini_infer/models/gpt_checkpoint.h"
#include "mini_infer/ops/loss_fn.h"
#include "mini_infer/optim/adam.h"
#include "mini_infer/tokenizer/tokenizer.h"
#include "utils/logger.h"
#ifdef ENABLE_CUDA
#include "runtime/cuda_check.h"
#endif

namespace {
using namespace mtensor;
using TokenIds = std::vector<std::int32_t>;

int ParsePositiveInt(std::string_view text) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value <= 0) {
        throw std::invalid_argument("Expected a positive integer: " + std::string(text));
    }
    return value;
}

Tensor WindowLoss(GPT& model, const TokenIds& tokens, std::size_t start, int length, Device device) {
    const Tensor input(tokens.data() + start, {length}, Dtype::kInt32, device);
    const Tensor target(tokens.data() + start + 1, {length}, Dtype::kInt32, device);
    return cross_entropy(model(input), target);
}

float ReadLoss(const Tensor& loss) {
    NogradNode no_grad;
    const Tensor host = loss.to(Device::kCPU);
    return *static_cast<const float*>(host.data_ptr());
}

double Evaluate(GPT& model, const TokenIds& tokens, std::size_t begin, std::size_t end,
                int length, Device device) {
    NogradNode no_grad;
    // 每次评估使用相同的窗口。
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> window(begin, end - length - 1);
    constexpr int samples = 16;
    double total = 0.0;
    for (int i = 0; i < samples; ++i) {
        total += ReadLoss(WindowLoss(model, tokens, window(rng), length, device));
    }
    return total / samples;
}

void EvaluateAndLog(Logger& training_log, GPT& model, const TokenIds& tokens, std::size_t train_end,
                    int context_length, Device device, int step) {
    const double train_loss = Evaluate(model, tokens, 0, train_end, context_length, device);
    const double val_loss = Evaluate(model, tokens, train_end, tokens.size(), context_length, device);
    if (!std::isfinite(train_loss) || !std::isfinite(val_loss)) {
        throw std::runtime_error("Evaluation loss is not finite");
    }
    std::ostringstream summary;
    summary << "eval step=" << step << " train_loss=" << train_loss << " val_loss=" << val_loss;
    training_log.info(summary.str());
    std::cout << summary.str() << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    using namespace mtensor;
    try {
        if (argc < 2 || argc > 5) {
            std::cerr << "Usage: train_gpt <corpus.txt> [steps] [cpu|cuda] [batch_size]\n";
            return 1;
        }
        const int steps = argc >= 3 ? ParsePositiveInt(argv[2]) : 1000;
        const Device device = GptDevice(argc, argv, 3);
        const char* backend = device == Device::kCUDA ? "cuda" : "cpu";
        const int batch_size = argc >= 5 ? ParsePositiveInt(argv[4]) : 2;

        std::ifstream file(argv[1], std::ios::binary);
        if (!file) throw std::runtime_error("Could not open corpus: " + std::string(argv[1]));
        const std::string corpus((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const Tokenizer tokenizer(corpus);
        const TokenIds tokens = tokenizer.encode(corpus);
        const GPTConfig config{tokenizer.vocab_size()};
        const std::size_t train_end = tokens.size() * 9 / 10;
        const auto context = static_cast<std::size_t>(config.context_length);
        if (train_end <= context || tokens.size() - train_end <= context) {
            throw std::invalid_argument("Training and validation splits both need more tokens than the context length");
        }

        GPT model(config, device);
        Adam optimizer(model.parameters(), 0.001f, 0.9f, 0.99f);
        const Tensor loss_scale = full({}, 1.0f / batch_size, Dtype::kFloat32, device);
        Logger training_log("train_gpt.log");
        std::size_t parameters = 0;
        for (const Tensor& parameter : model.parameters()) parameters += parameter.numel();
        std::ostringstream start;
        start << "device=" << backend << " parameters=" << parameters << " vocab=" << config.vocab_size
              << " context=" << config.context_length << " layers=" << config.layers
              << " heads=" << config.heads << " channels=" << config.channels << " batch_size=" << batch_size
              << " train_tokens=" << train_end << " val_tokens=" << tokens.size() - train_end;
        training_log.info(start.str());
        std::cout << start.str() << '\n';
        EvaluateAndLog(training_log, model, tokens, train_end, config.context_length, device, 0);

        std::mt19937 rng(1337);
        std::uniform_int_distribution<std::size_t> window(0, train_end - context - 1);
        for (int iteration = 0; iteration < steps; ++iteration) {
            const int step = iteration + 1;
            const auto begin = std::chrono::steady_clock::now();
            optimizer.zero_grad();
            double loss_sum = 0.0;
            for (int sequence = 0; sequence < batch_size; ++sequence) {
                Tensor loss = WindowLoss(model, tokens, window(rng), config.context_length, device);
                const float value = ReadLoss(loss);
                if (!std::isfinite(value)) throw std::runtime_error("Training loss is not finite");
                loss.backward(loss_scale);
                loss_sum += value;
            }
            optimizer.step();
#ifdef ENABLE_CUDA
            if (device == Device::kCUDA) CUDA_CHECK(cudaDeviceSynchronize());
#endif
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            std::ostringstream stats;
            stats << "step=" << step << '/' << steps << " loss=" << loss_sum / batch_size
                  << " elapsed_ms=" << elapsed_ms;
            training_log.info(stats.str());
            if (step == 1 || step % 20 == 0 || step == steps) std::cout << stats.str() << '\n';
            if (step == steps || step % 200 == 0) {
                EvaluateAndLog(training_log, model, tokens, train_end, config.context_length, device, step);
            }
        }
        SaveGPTCheckpoint("gpt_final.bin", model, tokenizer);
        training_log.info("saved checkpoint gpt_final.bin");
        std::cout << "saved checkpoint gpt_final.bin\n";
        const std::string prompt = corpus.substr(0, 10);
        std::mt19937 sample_rng(1338);
        const std::string sample = prompt + GenerateGPT(model, tokenizer, prompt, device, sample_rng);
        training_log.info("generated sample\n" + sample);
        std::cout << "generated sample\n" << sample << '\n';
    } catch (const std::exception& error) {
        std::cerr << "GPT training failed: " << error.what() << '\n';
        return 1;
    }
}
