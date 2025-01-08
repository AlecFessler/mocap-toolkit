// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef LOGGING_H
#define LOGGING_H

enum log_level {
  INFO,
  BENCHMARK,
  DEBUG,
  WARNING,
  ERROR,
  NUM_LOG_LEVELS
};

constexpr log_level LOG_LEVEL = BENCHMARK;

struct Logging {
private:
  Logging(const char* fpath);

public:
  static void setup_logging(const char* fpath);
  ~Logging();
};

#define log_(lvl, msg) \
  if constexpr(lvl >= LOG_LEVEL) \
    log_write(lvl, __FILE__, __LINE__, msg)

void log_write(
  log_level lvl,
  const char* file,
  int line,
  const char* log_str
);

#endif
