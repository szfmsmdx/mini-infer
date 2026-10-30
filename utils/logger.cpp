#include "logger.h"

#include <iostream>
#include <mutex>

namespace mtensor {

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
    static std::mutex output_mutex;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cerr << '[' << level << "] " << message << '\n';
}

} // namespace mtensor
