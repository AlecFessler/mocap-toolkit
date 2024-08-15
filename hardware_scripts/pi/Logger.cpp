#include "Logger.h"

#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <sstream>

Logger::Logger(const std::string& logFile) : logFile_(logFile, std::ios::app) {
  if (!logFile_.is_open())
    throw std::runtime_error("Could not open log file");
}

Logger::~Logger() {
  try {
    this->flush();
  } catch (...) {}
  logFile_.close();
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

static std::string logString(const std::string& timestamp, Logger::Level level, const std::string& file, int line, const std::string& message) {
  std::ostringstream oss;
  switch (level) {
    case Logger::Level::INFO:
      oss << timestamp << " [INFO] " << file << ":" << line << " - " << message << std::endl;
      return oss.str();
    case Logger::Level::WARNING:
      oss << timestamp << " [WARNING] " << file << ":" << line << " - " << message << std::endl;
      return oss.str();
    case Logger::Level::ERROR:
      oss << timestamp << " [ERROR] " << file << ":" << line << " - " << message << std::endl;
      return oss.str();
    default:
      oss << timestamp << " [UNKNOWN] " << file << ":" << line << " - " << message << std::endl;
      return oss.str();
  }
}

void Logger::log(const std::string& timestamp, Level level, const std::string& file, int line, const std::string& message) {
  this->flush();
  logFile_ << logString(timestamp, level, file, line, message);
}

void Logger::queue(const std::string& timestamp, Level level, const std::string& file, int line, const std::string& message) {
  queue_.push(logString(timestamp, level, file, line, message));
}

void Logger::flush() {
  while (!queue_.empty()) {
    logFile_ << queue_.front();
    queue_.pop();
  }
}
