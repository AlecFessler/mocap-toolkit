#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <string>

#define LOG_EVENT(timestamp, event) \
  "Event: " + event + " at: " + timestamp

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
  void log(const std::string& message, Level level);

  ~Logger();
  Logger(Logger const&) = delete;
  void operator=(Logger const&) = delete;
  Logger(Logger&&) = delete;
  void operator=(Logger&&) = delete;
};

#endif // LOGGER_H
