#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "gpt_generate.h"
#include "mini_infer/models/gpt_checkpoint.h"

int main(int argc, char** argv) {
    using namespace mtensor;
    try {
        if (argc < 2 || argc > 4) {
            std::cerr << "Usage: chat_gpt <checkpoint> [cpu|cuda] [prompt]\n";
            return 1;
        }
        const Device device = GptDevice(argc, argv, 2);
        auto [model, tokenizer] = LoadGPTCheckpoint(argv[1], device);
        std::mt19937 rng(1338);
        if (argc == 4) {
            std::cout << GenerateGPT(model, tokenizer, argv[3], device, rng) << '\n';
            return 0;
        }

        const auto context = static_cast<std::size_t>(model.config().context_length);
        std::string history;
        std::cout << "Enter an English prompt. Type /quit to exit.\n";
        for (std::string prompt; std::cout << "> " && std::getline(std::cin, prompt);) {
            if (prompt == "/quit") break;
            if (prompt.empty()) continue;
            try {
                const std::string input = history + prompt;
                const std::string generated = GenerateGPT(model, tokenizer, input, device, rng);
                history = input + generated;
                if (history.size() > context) history.erase(0, history.size() - context);
                std::cout << generated << '\n';
            } catch (const std::invalid_argument& error) {
                std::cerr << error.what() << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "GPT generation failed: " << error.what() << '\n';
        return 1;
    }
}
