#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "utils/logger.h"

namespace {

void Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    Check(input.is_open(), "Could not open log file for verification");
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

void TestSeparateFiles() {
    const std::filesystem::path directory = "logger_test_output";
    const std::filesystem::path engine_path = directory / "engine.log";
    const std::filesystem::path worker_path = directory / "worker.log";
    std::filesystem::remove(engine_path);
    std::filesystem::remove(worker_path);

    {
        mtensor::Logger engine(engine_path.string());
        mtensor::Logger worker(worker_path.string());
        engine.info("engine started");
        worker.warn("worker waiting");
        engine.error("engine stopped");
    }

    Check(ReadFile(engine_path) == "[INFO] engine started\n[ERROR] engine stopped\n",
          "Engine log contains wrong lines");
    Check(ReadFile(worker_path) == "[WARN] worker waiting\n",
          "Worker log contains wrong lines");

    {
        mtensor::Logger engine(engine_path.string());
        engine.info("engine restarted");
    }
    Check(ReadFile(engine_path) ==
              "[INFO] engine started\n[ERROR] engine stopped\n[INFO] engine restarted\n",
          "Reopening a log file must append");

    std::filesystem::remove(engine_path);
    std::filesystem::remove(worker_path);
    std::filesystem::remove(directory);
}

void TestInvalidPath() {
    bool threw = false;
    try {
        mtensor::Logger invalid("");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "Empty log path must be rejected");
}

void TestDefaultLogger() {
    const std::filesystem::path original = std::filesystem::current_path();
    const std::filesystem::path directory = original / "logger_test_output";
    std::filesystem::create_directory(directory);
    std::filesystem::current_path(directory);
    try {
        mtensor::Logger default_logger;
        default_logger.info("default logger");
    } catch (...) {
        std::filesystem::current_path(original);
        throw;
    }
    std::filesystem::current_path(original);
    Check(ReadFile(directory / ".log") == "[INFO] default logger\n",
          "Default logger must write to .log in the working directory");
    std::filesystem::remove(directory / ".log");
    std::filesystem::remove(directory);
}

}  // namespace

int main() {
    try {
        TestSeparateFiles();
        TestInvalidPath();
        TestDefaultLogger();
        std::cout << "Logger tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
