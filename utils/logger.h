#pragma once

#include <string>

namespace mtensor {

class Logger {
public:
    static void info(const std::string& message);
    static void warn(const std::string& message);
    static void error(const std::string& message);

private:
    static void write(const char* level, const std::string& message);
};

} // namespace mtensor
