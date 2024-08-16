#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <queue>
#include <string>

class Logger {
private:
  int fd_;

public:
  enum Level {
    INFO,
    WARNING,
    ERROR
  };

  void log(Level level, const char* file, int line, const char* message);

  Logger(const char* filename);
  ~Logger();
  Logger(Logger const&) = delete;
  void operator=(Logger const&) = delete;
  Logger(Logger&&) = delete;
  void operator=(Logger&&) = delete;
};

#endif // LOGGER_H
