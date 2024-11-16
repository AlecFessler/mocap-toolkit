// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <queue>
#include <string>

class logger_t {
private:
  int fd_;

public:
  enum level_t {
    INFO,
    DEBUG,
    WARNING,
    ERROR
  };

  void log(level_t level, const char* file, int line, const char* message);

  logger_t(const char* filename);
  ~logger_t();
  logger_t(logger_t const&) = delete;
  void operator=(logger_t const&) = delete;
  logger_t(logger_t&&) = delete;
  void operator=(logger_t&&) = delete;
};

#endif // LOGGER_H
