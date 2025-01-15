#ifndef LOGGING_H
#define LOGGING_H

#define log(lvl, msg) log_msg(lvl, __FILE__, __LINE__, msg)

typedef enum log_level {
  INFO,
  BENCHMARK,
  DEBUG,
  WARNING,
  ERROR,
  NUM_LOG_LEVELS
} log_level;

int setup_logging(const char* fpath);
void cleanup_logging();
void log_msg(log_level lvl, const char* file, int line, const char* log_str);

#endif
