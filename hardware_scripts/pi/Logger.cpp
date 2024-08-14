#include "Logger.h"
#include <stdexcept>

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

void Logger::log(const std::string& message, Level level) {
  switch (level) {
    case Level::INFO:
      logFile_ << "[INFO] " << message << std::endl;
      break;
    case Level::WARNING:
      logFile_ << "[WARNING] " << message << std::endl;
      break;
    case Level::ERROR:
      logFile_ << "[ERROR] " << message << std::endl;
      break;
    default:
      logFile_ << "[UNKNOWN] " << message << std::endl;
      break;
  }
}
