#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <string>

class Logger {
private:
  static Logger* instance_;
  std::ofstream logFile_;

  Logger(const std::string& logFile);

public:
  enum Level {
    INFO,
    WARNING,
    ERROR
  };

  static Logger& getLogger(const std::string& logFile = "log.txt");
  static std::string timestamp();
  void log(const std::string& timestamp, Level level, const std::string& file, int line, const std::string& message);

  ~Logger();
  Logger(Logger const&) = delete;
  void operator=(Logger const&) = delete;
  Logger(Logger&&) = delete;
  void operator=(Logger&&) = delete;
};

#endif // LOGGER_H
