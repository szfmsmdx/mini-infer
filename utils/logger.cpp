#include <filesystem>
#include <ostream>
#include <stdexcept>

#include "utils/logger.h"

namespace mtensor {

Logger::Logger(const std::string& file_path) : file_path_(file_path) {
    if (file_path.empty()) {
        throw std::invalid_argument("Logger file path must not be empty");
    }
}

void Logger::info(const std::string& message) {
    write("INFO", message);
}

void Logger::warn(const std::string& message) {
    write("WARN", message);
}

void Logger::error(const std::string& message) {
    write("ERROR", message);
}

void Logger::write(const char* level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_.is_open()) {
        const std::filesystem::path path(file_path_);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        file_.open(path, std::ios::app);
        if (!file_) {
            throw std::runtime_error("Cannot open log file: " + file_path_);
        }
    }
    file_ << '[' << level << "] " << message << std::endl;
    if (!file_) {
        throw std::runtime_error("Logger output failed");
    }
}

}  // namespace mtensor
