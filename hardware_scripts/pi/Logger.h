#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <queue>
#include <string>

class Logger {
private:
  std::ofstream logFile_;
  std::queue<std::string> queue_;

public:
  enum Level {
    INFO,
    WARNING,
    ERROR
  };

  static std::string timestamp();
  void log(const std::string& timestamp, Level level, const std::string& file, int line, const std::string& message);
  void queue(const std::string& timestamp, Level level, const std::string& file, int line, const std::string& message);
  void flush();

  Logger(const std::string& logFile);
  ~Logger();
  Logger(Logger const&) = delete;
  void operator=(Logger const&) = delete;
  Logger(Logger&&) = delete;
  void operator=(Logger&&) = delete;
};

#endif // LOGGER_H
