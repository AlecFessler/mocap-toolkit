#include "Logger.h"
#include <stdexcept>

Logger* Logger::instance_ = nullptr;

Logger::Logger(const std::string& logFile) : logFile_(logFile) {
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

void Logger::log(const std::string& message, LogLevel level) {
  switch (level) {
    case LogLevel::INFO:
      logFile_ << "[INFO] " << message << std::endl;
      break;
    case LogLevel::WARNING:
      logFile_ << "[WARNING] " << message << std::endl;
      break;
    case LogLevel::ERROR:
      logFile_ << "[ERROR] " << message << std::endl;
      break;
    default:
      logFile_ << "[UNKNOWN] " << message << std::endl;
      break;
  }
}
