#include "Logger.h"

#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <sstream>

Logger* Logger::instance_ = nullptr;

Logger::Logger(const std::string& logFile) : logFile_(logFile, std::ios::app) {
  if (!logFile_.is_open())
    throw std::runtime_error("Could not open log file");
}

Logger::~Logger() {
  logFile_.close();
}

Logger& Logger::getLogger(const std::string& logFile) {
  if (instance_ == nullptr)
    instance_ = new Logger(logFile);
  return *instance_;
}

std::string Logger::timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto duration = now.time_since_epoch();
    auto uS = std::chrono::duration_cast<std::chrono::microseconds>(duration) % std::chrono::seconds(1);
    std::tm now_tm = *std::gmtime(&now_time_t);
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S")
        << '.' << std::setw(6) << std::setfill('0') << uS.count() << 'Z';
    return oss.str();
}

void Logger::log(const std::string& timestamp, Level level, const std::string& file, int line, const std::string& message) {
  switch (level) {
    case Level::INFO:
      logFile_ << timestamp << " [INFO] " << file << ":" << line << " - " << message << std::endl;
      break;
    case Level::WARNING:
      logFile_ << timestamp << " [WARNING] " << file << ":" << line << " - " << message << std::endl;
      break;
    case Level::ERROR:
      logFile_ << timestamp << " [ERROR] " << file << ":" << line << " - " << message << std::endl;
      break;
    default:
      logFile_ << timestamp << " [UNKNOWN] " << file << ":" << line << " - " << message << std::endl;
      break;
  }
}
