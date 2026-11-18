#pragma once

#include <fstream>
#include <mutex>
#include <string>

namespace mtensor {

class Logger {
public:
    Logger() = default;
    explicit Logger(const std::string& file_path);

    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);

private:
    void write(const char* level, const std::string& message);

    std::string file_path_ = ".log";
    std::ofstream file_;
    std::mutex mutex_;
};

inline Logger logger;

}  // namespace mtensor
